import numpy as np
import scipy.sparse as sp

import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.init import xavier_uniform_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType
from KGCN import KGCN

import utils
from spmm import SpecialSpmm, CHUNK_SIZE_FOR_SPMM


class NIKG(KnowledgeRecommender):
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(NIKG, self).__init__(config, dataset)
        # load base para
        self.embedding_size = config['embedding_size']
        self.kg_embedding_size = config['kg_embedding_size']
        self.reg_weight = config['reg_weight']
        self.prune_threshold = config['prune_threshold']
        self.n_layers = config['n_layers']
        self.neighbor_sample_size = config["neighbor_sample_size"]
        
        # define layers
        self.user_embeddings = nn.Embedding(self.n_users, self.embedding_size)
        self.item_embeddings = nn.Embedding(self.n_items, self.embedding_size)
        # self.relation_embeddings = nn.Embedding(self.n_relations, self.kg_embedding_size)
        self.entity_embeddings = nn.Embedding(self.n_entities, self.embedding_size)

        # generate user-item interaction_matrix
        self.inter_matrix_type = config['inter_matrix_type']
        value_field = self.RATING if self.inter_matrix_type == 'rating' else None
        self.interaction_matrix = dataset.inter_matrix(form='coo', value_field=value_field).astype(np.float32)
        self.adj_matrix = self.get_adj_mat()
        self.norm_adj_matrix = self.get_norm_mat().to(self.device)
        
        # generate kg interaction matrix
        self.matrix_size = torch.Size(
            [self.n_entities, self.n_entities]
        )
        kg_graph = dataset.kg_graph(form="coo", value_field="relation_id")
        self.dgl_kg_graph = dataset.kg_graph(form="dgl", value_field="relation_id")
        self.adj_kg = (
            self.construct_kg_adj(self.dgl_kg_graph)
        )
        adj_entity, adj_relation = self.construct_adj(kg_graph)
        self.adj_entity, self.adj_relation = adj_entity.to(
            self.device
        ), adj_relation.to(self.device)

        # for learn adj
        self.spmm = config['spmm']
        self.special_spmm = SpecialSpmm() if self.spmm == 'spmm' else torch.sparse.mm

        # init current kg-based algorithms
        self.kg_model = KGCN(config, dataset)
        
        self.bce_loss = nn.BCEWithLogitsLoss()
        self.l2_loss = EmbLoss()
        self.restore_user_e = None
        self.restore_entity_e = None

        self.apply(xavier_uniform_initialization)
        self.other_parameter_name = ["restore_user_e", "restore_entity_e"]
    
    def construct_adj(self, kg_graph):
        r"""Get neighbors and corresponding relations for each entity in the KG.

        Args:
            kg_graph(scipy.sparse.coo_matrix): an undirected graph

        Returns:
            tuple:
                - adj_entity(torch.LongTensor): each line stores the sampled neighbor entities for a given entity,
                  shape: [n_entities, neighbor_sample_size]
                - adj_relation(torch.LongTensor): each line stores the corresponding sampled neighbor relations,
                  shape: [n_entities, neighbor_sample_size]
        """
        # self.logger.info('constructing knowledge graph ...')
        # treat the KG as an undirected graph
        kg_dict = dict()
        for triple in zip(kg_graph.row, kg_graph.data, kg_graph.col):
            head = triple[0]
            relation = triple[1]
            tail = triple[2]
            if head not in kg_dict:
                kg_dict[head] = []
            kg_dict[head].append((tail, relation))
            if tail not in kg_dict:
                kg_dict[tail] = []
            kg_dict[tail].append((head, relation))

        # self.logger.info('constructing adjacency matrix ...')
        # each line of adj_entity stores the sampled neighbor entities for a given entity
        # each line of adj_relation stores the corresponding sampled neighbor relations
        entity_num = kg_graph.shape[0]
        adj_entity = np.zeros([entity_num, self.neighbor_sample_size], dtype=np.int64)
        adj_relation = np.zeros([entity_num, self.neighbor_sample_size], dtype=np.int64)
        for entity in range(entity_num):
            if entity not in kg_dict.keys():
                adj_entity[entity] = np.array([entity] * self.neighbor_sample_size)
                adj_relation[entity] = np.array([0] * self.neighbor_sample_size)
                continue

            neighbors = kg_dict[entity]
            n_neighbors = len(neighbors)
            if n_neighbors >= self.neighbor_sample_size:
                sampled_indices = np.random.choice(
                    list(range(n_neighbors)),
                    size=self.neighbor_sample_size,
                    replace=False,
                )
            else:
                sampled_indices = np.random.choice(
                    list(range(n_neighbors)),
                    size=self.neighbor_sample_size,
                    replace=True,
                )
            adj_entity[entity] = np.array([neighbors[i][0] for i in sampled_indices])
            adj_relation[entity] = np.array([neighbors[i][1] for i in sampled_indices])

        return torch.from_numpy(adj_entity), torch.from_numpy(adj_relation)
    
    def construct_kg_adj(self, kg_graph):
        import dgl
        
        r"""Get the initial weight matrix through the knowledge graph

        Returns:
            torch.sparse.FloatTensor: Sparse tensor of the attention matrix
        """
        adj_list = []
        for rel_type in range(1, self.n_relations, 1):
            edge_idxs = kg_graph.filter_edges(
                lambda edge: edge.data["relation_id"] == rel_type
            )
            sub_graph = (
                dgl.edge_subgraph(kg_graph, edge_idxs, relabel_nodes=False)
                .adj_external(transpose=False, scipy_fmt="coo")
                .astype("float")
            )
            rowsum = np.array(sub_graph.sum(1))
            d_inv = np.power(rowsum, -1).flatten()
            d_inv[np.isinf(d_inv)] = 0.0
            d_mat_inv = sp.diags(d_inv)
            norm_adj = d_mat_inv.dot(sub_graph).tocoo()
            adj_list.append(norm_adj)

        final_adj_matrix = sum(adj_list).tocoo()
        indices = torch.LongTensor([final_adj_matrix.row, final_adj_matrix.col])
        values = torch.FloatTensor(final_adj_matrix.data)
        adj_matrix_tensor = torch.sparse.FloatTensor(indices, values, self.matrix_size)
        return adj_matrix_tensor.to(self.device)
    
    # Generate adj
    def get_adj_mat(self, data=None):
        if data is None:
            data = [1] * self.interaction_matrix.data
        inter_M_t = self.interaction_matrix.transpose()
        A = sp.dok_matrix((self.n_users + self.n_items, self.n_users + self.n_items), dtype=np.float32)
        data_dict = dict(zip(zip(self.interaction_matrix.row, self.interaction_matrix.col + self.n_users), data))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), data)))
        A._update(data_dict)  # dok_matrix
        return A

    def get_norm_mat(self):
        r""" A_{hat} = D^{-0.5} \times A \times D^{-0.5} """
        # norm adj matrix
        sumArr = (self.adj_matrix > 0).sum(axis=1)
        # add epsilon to avoid divide by zero Warning
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * self.adj_matrix * D
        # covert norm_adj matrix to tensor
        SparseL = utils.sp2tensor(L)
        return SparseL

    # Learn adj
    def sp_cos_sim(self, a, b, eps=1e-8, CHUNK_SIZE=CHUNK_SIZE_FOR_SPMM):
        self.adj_indices = self.adj_kg.coalesce().indices()
        self.adj_shape = self.adj_kg.coalesce().shape
        a_n, b_n = a.norm(dim=1)[:, None], b.norm(dim=1)[:, None]
        a_norm = a / torch.max(a_n, eps * torch.ones_like(a_n))
        b_norm = b / torch.max(b_n, eps * torch.ones_like(b_n))

        L = self.adj_indices.shape[1]
        sims = torch.zeros(L, dtype=a.dtype).to(self.device)
        for idx in range(0, L, CHUNK_SIZE):
            batch_indices = self.adj_indices[:, idx:idx + CHUNK_SIZE]

            a_batch = torch.index_select(a_norm, 0, batch_indices[0, :])
            b_batch = torch.index_select(b_norm, 0, batch_indices[1, :])

            dot_prods = torch.mul(a_batch, b_batch).sum(1)
            sims[idx:idx + CHUNK_SIZE] = dot_prods

        return torch.sparse_coo_tensor(self.adj_indices, sims, size=self.adj_kg.shape,
                                       dtype=sims.dtype).coalesce()

    def get_sim_mat(self, entity_feature):
        # user_feature = self.get_all_user_embedding().to(self.device)
        # item_feature = self.get_all_item_embedding().to(self.device)
        sim_inter = self.sp_cos_sim(entity_feature, entity_feature)
        return sim_inter

    def inter2adj(self, inter):
        adj_data = inter.values()
        adj = torch.sparse.FloatTensor(self.adj_indices, adj_data, self.adj_shape).to(self.device).coalesce()
        return adj

    def get_sim_adj(self, pruning, entity_feature):
        sim_mat = self.get_sim_mat(entity_feature)
        sim_adj = self.inter2adj(sim_mat)

        # pruning
        sim_value = torch.div(torch.add(sim_adj.values(), 1), 2)
        pruned_sim_value = torch.where(sim_value < pruning, torch.zeros_like(sim_value),
                                       sim_value) if pruning > 0 else sim_value
        pruned_sim_adj = torch.sparse.FloatTensor(sim_adj.indices(), pruned_sim_value, self.adj_shape).coalesce()
        self.pruned_sim_adj = pruned_sim_adj

        # normalize
        pruned_sim_indices = pruned_sim_adj.indices()
        diags = torch.sparse.sum(pruned_sim_adj, dim=1).to_dense() + 1e-7
        diags = torch.pow(diags, -1)
        diag_lookup = diags[pruned_sim_indices[0, :]]

        pruned_sim_adj_value = pruned_sim_adj.values()
        normal_sim_value = torch.mul(pruned_sim_adj_value, diag_lookup)
        normal_sim_adj = torch.sparse.FloatTensor(pruned_sim_indices, normal_sim_value,
                                                  self.adj_shape).to(self.device).coalesce()

        return normal_sim_adj

    def update_entity_embs(self, entity_feature, item_embeddings):
        update_index = torch.arange(self.n_items).to(self.device)
        ori_entity_embs = entity_feature[update_index]
        comb_entity_embs = torch.stack([ori_entity_embs, item_embeddings], dim=1)
        entity_feature[update_index] = torch.mean(comb_entity_embs, dim=1)

        return entity_feature
    
    def adj_combine(self, kg_adj_local, kg_adj_cf):
        # kg_adj_local & kg_adj_cf: if one of the matrix element is zero the correspond position value will be zero.
        adj_local_values = kg_adj_local.values()
        adj_cf_values = kg_adj_cf.values()
        # Create masks for each tensor which values are equal with 0
        local_mask = torch.eq(adj_local_values, 0)
        cf_mask = torch.eq(adj_cf_values, 0)
        mask = torch.logical_or(local_mask, cf_mask)
        
        # calculate the noisy edge rate
        # zero_v = torch.zeros_like(adj_cf_values)
        # zero_v_len = zero_v.shape[0]
        # cf_zero_num = (adj_cf_values == zero_v).sum()
        # local_zero_num = (adj_local_values == zero_v).sum()
        # cf_prun_rate = torch.mul(cf_zero_num/zero_v_len, 100) # prun:0.6--95.68, 0.28--0.0092, 0.29--0.0161, 0.39--1.7822
        # local_prun_rate = torch.mul(local_zero_num/zero_v_len, 100) # 0.6--61.63 0.28--0.0069, 0.29--0.0138, 0.39--1.8075

        # Calculate the mean of the original tensors with 0 values
        kg_adj = (kg_adj_local + kg_adj_cf)/2
        kg_adj = kg_adj.coalesce()
        indices = kg_adj.indices()
        # values = kg_adj.values()
        indices_mask = torch.broadcast_to(torch.reshape(mask, (1, -1)), indices.size())
        zero_indices = torch.empty(2, 0).to(self.device)
        zero_indices = torch.reshape(indices[indices_mask], (2, -1))
        # zero_values = torch.zeros_like(values[mask])
        # filter_mask = torch.logical_not(mask)
        # filter_indices_mask = torch.broadcast_to(torch.reshape(filter_mask, (1, -1)), indices.size())
        # filter_indices = torch.reshape(indices[filter_indices_mask], (2, -1))
        # filter_values = values[filter_mask]
        # filter_values = torch.cat((filter_values, zero_values), 0)
        # filter_indices = torch.cat((filter_indices, zero_indices), 1)
        # combine_kg_adj = torch.sparse.FloatTensor(filter_indices, filter_values,
        #                                           self.adj_shape).to(self.device)
        # combine_kg_adj = combine_kg_adj.coalesce()

        return zero_indices
    
    def only_local_adj(self, kg_adj_local, kg_adj_cf):
        # kg_adj_local & kg_adj_cf: if one of the matrix element is zero the correspond position value will be zero.
        adj_local_values = kg_adj_local.values()
        # adj_cf_values = kg_adj_cf.values()
        # Create masks for each tensor which values are equal with 0
        local_mask = torch.eq(adj_local_values, 0)
        mask = local_mask

        # Calculate the mean of the original tensors with 0 values
        kg_adj = kg_adj_local
        kg_adj = kg_adj.coalesce()
        indices = kg_adj.indices()
        # values = kg_adj.values()
        indices_mask = torch.broadcast_to(torch.reshape(mask, (1, -1)), indices.size())
        zero_indices = torch.empty(2, 0).to(self.device)
        zero_indices = torch.reshape(indices[indices_mask], (2, -1))

        return zero_indices
    
    def only_cf_adj(self, kg_adj_local, kg_adj_cf):
        # kg_adj_local & kg_adj_cf: if one of the matrix element is zero the correspond position value will be zero.
        adj_cf_values = kg_adj_cf.values()
        # Create masks for each tensor which values are equal with 0
        cf_mask = torch.eq(adj_cf_values, 0)
        mask = cf_mask

        # Calculate the mean of the original tensors with 0 values
        kg_adj = kg_adj_cf
        kg_adj = kg_adj.coalesce()
        indices = kg_adj.indices()
        # values = kg_adj.values()
        indices_mask = torch.broadcast_to(torch.reshape(mask, (1, -1)), indices.size())
        zero_indices = torch.empty(2, 0).to(self.device)
        zero_indices = torch.reshape(indices[indices_mask], (2, -1))

        return zero_indices
    
    # Train KGCN
    def forward(self, user, item, pruning=0.0):
        # local structure-based measurement
        entity_feature = self.entity_embeddings.weight
        entity_embs_list = [entity_feature]
        for _ in range(self.n_layers):
            entity_feature = self.special_spmm(self.adj_kg, entity_feature)
            entity_embs_list.append(entity_feature)
        
        entity_embs = torch.stack(entity_embs_list, dim=1)
        entity_embs = torch.mean(entity_embs, dim=1)
        
        kg_adj_local = self.get_sim_adj(pruning, entity_embs)

        # CF-based measurement
        user_embeddings = self.user_embeddings.weight
        item_embeddings = self.item_embeddings.weight
        ui_embeddings = torch.cat([user_embeddings, item_embeddings], dim=0)
        embeddings_list = [ui_embeddings]
        for _ in range(self.n_layers):
            ui_embeddings = self.special_spmm(self.norm_adj_matrix, ui_embeddings)
            embeddings_list.append(ui_embeddings)

        lightgcn_ui_embeddings = torch.stack(embeddings_list, dim=1)
        lightgcn_ui_embeddings = torch.mean(lightgcn_ui_embeddings, dim=1)

        _, item_all_embeddings = torch.split(lightgcn_ui_embeddings, [self.n_users, self.n_items])
        cf_entity_feature = self.update_entity_embs(entity_feature, item_all_embeddings)
        kg_adj_cf = self.get_sim_adj(pruning, cf_entity_feature)
        
        # drop nodes and edges
        zero_indices = self.adj_combine(kg_adj_local, kg_adj_cf)
        mask = torch.zeros_like(self.adj_entity, dtype=torch.bool)
        value_mask = zero_indices[1, :].unsqueeze(1).expand(-1, self.adj_entity.shape[1])
        mask[zero_indices[0, :]] = self.adj_entity[zero_indices[0, :]] == value_mask
        # for key, value in zip(zero_indices[0, :], zero_indices[1, :]):
        #     # Check if the value is in the corresponding row of d2
        #     mask[key, self.adj_entity[key] == value] = True
        row_indices = torch.nonzero(mask)[:, 0]
        self.adj_entity[mask] = row_indices
        self.adj_relation[mask] = 0

        user_all_embeddings, item_all_embeddings = self.kg_model(self.adj_entity, self.adj_relation, self.user_embeddings, self.entity_embeddings, user, item)

        return user_all_embeddings, item_all_embeddings
    
    # KGCN
    def calculate_loss(self, interaction):
        user = interaction[self.USER_ID]
        pos_item = interaction[self.ITEM_ID]
        neg_item = interaction[self.NEG_ITEM_ID]

        user_e, pos_item_e = self.forward(user, pos_item, pruning=self.prune_threshold, )
        user_e, neg_item_e = self.forward(user, neg_item, pruning=self.prune_threshold, )

        pos_item_score = torch.mul(user_e, pos_item_e).sum(dim=1)
        neg_item_score = torch.mul(user_e, neg_item_e).sum(dim=1)

        predict = torch.cat((pos_item_score, neg_item_score))
        target = torch.zeros(len(user) * 2, dtype=torch.float32).to(self.device)
        target[: len(user)] = 1
        rec_loss = self.bce_loss(predict, target)

        l2_loss = self.l2_loss(user_e, pos_item_e, neg_item_e)
        loss = rec_loss + self.reg_weight * l2_loss

        return loss
    
    def predict(self, interaction):
        user = interaction[self.USER_ID]
        item = interaction[self.ITEM_ID]
        user_e, item_e = self.forward(user, item)
        return torch.mul(user_e, item_e).sum(dim=1)

    def full_sort_predict(self, interaction):
        user_index = interaction[self.USER_ID]
        item_index = torch.tensor(range(self.n_items)).to(self.device)

        user = torch.unsqueeze(user_index, dim=1).repeat(1, item_index.shape[0])
        user = torch.flatten(user)
        item = torch.unsqueeze(item_index, dim=0).repeat(user_index.shape[0], 1)
        item = torch.flatten(item)

        user_e, item_e = self.forward(user, item)
        score = torch.mul(user_e, item_e).sum(dim=1)

        return score.view(-1)