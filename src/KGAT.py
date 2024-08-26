# -*- coding: utf-8 -*-
# @Time   : 2020/9/15
# @Author : Shanlei Mu
# @Email  : slmu@ruc.edu.cn

r"""
KGAT
##################################################
Reference:
    Xiang Wang et al. "KGAT: Knowledge Graph Attention Network for Recommendation." in SIGKDD 2019.

Reference code:
    https://github.com/xiangwang1223/knowledge_graph_attention_network
"""

import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import BPRLoss, EmbLoss
from recbole.utils import InputType


class Aggregator(nn.Module):
    """GNN Aggregator layer"""

    def __init__(self, input_dim, output_dim, dropout, aggregator_type):
        super(Aggregator, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.aggregator_type = aggregator_type

        self.message_dropout = nn.Dropout(dropout)

        if self.aggregator_type == "gcn":
            self.W = nn.Linear(self.input_dim, self.output_dim)
        elif self.aggregator_type == "graphsage":
            self.W = nn.Linear(self.input_dim * 2, self.output_dim)
        elif self.aggregator_type == "bi":
            self.W1 = nn.Linear(self.input_dim, self.output_dim)
            self.W2 = nn.Linear(self.input_dim, self.output_dim)
        else:
            raise NotImplementedError

        self.activation = nn.LeakyReLU()

    def forward(self, norm_matrix, ego_embeddings):
        side_embeddings = torch.sparse.mm(norm_matrix, ego_embeddings)

        if self.aggregator_type == "gcn":
            ego_embeddings = self.activation(self.W(ego_embeddings + side_embeddings))
        elif self.aggregator_type == "graphsage":
            ego_embeddings = self.activation(
                self.W(torch.cat([ego_embeddings, side_embeddings], dim=1))
            )
        elif self.aggregator_type == "bi":
            add_embeddings = ego_embeddings + side_embeddings
            sum_embeddings = self.activation(self.W1(add_embeddings))
            bi_embeddings = torch.mul(ego_embeddings, side_embeddings)
            bi_embeddings = self.activation(self.W2(bi_embeddings))
            ego_embeddings = bi_embeddings + sum_embeddings
        else:
            raise NotImplementedError

        ego_embeddings = self.message_dropout(ego_embeddings)

        return ego_embeddings


class KGAT(KnowledgeRecommender):
    r"""KGAT is a knowledge-based recommendation model. It combines knowledge graph and the user-item interaction
    graph to a new graph called collaborative knowledge graph (CKG). This model learns the representations of users and
    items by exploiting the structure of CKG. It adopts a GNN-based architecture and define the attention on the CKG.
    """

    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(KGAT, self).__init__(config, dataset)
        # load dataset info
        self.ckg = dataset.ckg_graph(form="dgl", value_field="relation_id")
        self.all_hs = torch.LongTensor(
            dataset.ckg_graph(form="coo", value_field="relation_id").row
        ).to(self.device)
        self.all_ts = torch.LongTensor(
            dataset.ckg_graph(form="coo", value_field="relation_id").col
        ).to(self.device)
        self.all_rs = torch.LongTensor(
            dataset.ckg_graph(form="coo", value_field="relation_id").data
        ).to(self.device)
        self.matrix_size = torch.Size(
            [self.n_users + self.n_entities, self.n_users + self.n_entities]
        )

        # load parameters info
        self.embedding_size = config["embedding_size"]
        self.kg_embedding_size = config["kg_embedding_size"]
        self.layers = [self.embedding_size] + config["layers"]
        self.aggregator_type = config["aggregator_type"]
        self.mess_dropout = config["mess_dropout"]
        self.reg_weight = config["reg_weight"]

        # generate intermediate data
        self.A_in = (
            self.init_graph()
        )  # init the attention matrix by the structure of ckg

        # define layers and loss
        # self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        # self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_embedding = nn.Embedding(self.n_relations, self.kg_embedding_size)
        self.trans_w = nn.Embedding(
            self.n_relations, self.embedding_size * self.kg_embedding_size
        )
        
        self.aggregator_layers = nn.ModuleList()
        for idx, (input_dim, output_dim) in enumerate(
            zip(self.layers[:-1], self.layers[1:])
        ):
            self.aggregator_layers.append(
                Aggregator(
                    input_dim, output_dim, self.mess_dropout, self.aggregator_type
                )
            )
        self.mf_loss = BPRLoss()
        self.reg_loss = EmbLoss()
        self.tanh = nn.Tanh()
        # self.restore_user_e = None
        # self.restore_entity_e = None

        # parameters initialization
        self.apply(xavier_normal_initialization)
        # self.other_parameter_name = ["restore_user_e", "restore_entity_e"]

    def init_graph(self):
        r"""Get the initial attention matrix through the collaborative knowledge graph

        Returns:
            torch.sparse.FloatTensor: Sparse tensor of the attention matrix
        """
        import dgl

        adj_list = []
        for rel_type in range(1, self.n_relations, 1):
            edge_idxs = self.ckg.filter_edges(
                lambda edge: edge.data["relation_id"] == rel_type
            )
            sub_graph = (
                dgl.edge_subgraph(self.ckg, edge_idxs, relabel_nodes=False)
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

    def _get_ego_embeddings(self, user_embedding, entity_embedding):
        user_embeddings = user_embedding.weight
        entity_embeddings = entity_embedding.weight
        ego_embeddings = torch.cat([user_embeddings, entity_embeddings], dim=0)
        return ego_embeddings
    
    def _get_update_A(self, combine_kg_adj):
        combine_kg_adj_v = combine_kg_adj.values()
        combine_kg_adj_i = combine_kg_adj.indices()
        zero_v_mask = torch.eq(combine_kg_adj_v, 0)
        zero_i_mask = torch.broadcast_to(torch.reshape(zero_v_mask, (1, -1)), combine_kg_adj_i.size())
        zero_indices = torch.empty(2, 0).to(self.device)
        zero_indices = torch.reshape(combine_kg_adj_i[zero_i_mask], (2, -1))
        zero_indices_src = zero_indices[0, :] + self.n_users
        zero_indices_tgt = zero_indices[1, :] + self.n_users
        self.A_in = self.A_in.coalesce()
        A_indices = self.A_in.indices()
        A_indices_src = A_indices[0, :]
        A_indices_tgt = A_indices[1, :]
        A_values = self.A_in.values()
        A_mask_src = torch.isin(A_indices_src, zero_indices_src)
        A_mask_tgt = torch.isin(A_indices_tgt, zero_indices_tgt)
        A_mask = A_mask_src & A_mask_tgt
        A_mask_i = torch.broadcast_to(torch.reshape(A_mask, (1, -1)), A_indices.size())
        update_A_mask = torch.logical_not(A_mask)
        update_A_mask_i = torch.logical_not(A_mask_i)
        update_A_indices = torch.reshape(A_indices[update_A_mask_i], (2, -1))
        update_A_values = A_values[update_A_mask]
        self.A_in = torch.sparse.FloatTensor(update_A_indices, update_A_values,
                                                  self.matrix_size).to(self.device).coalesce()

    def forward(self, combine_kg_adj, user_embedding, entity_embedding):
        ego_embeddings = self._get_ego_embeddings(user_embedding, entity_embedding)
        embeddings_list = [ego_embeddings]
        self._get_update_A(combine_kg_adj)
        
        for aggregator in self.aggregator_layers:
            ego_embeddings = aggregator(self.A_in, ego_embeddings)
            norm_embeddings = F.normalize(ego_embeddings, p=2, dim=1)
            embeddings_list.append(norm_embeddings)
        kgat_all_embeddings = torch.cat(embeddings_list, dim=1)
        user_all_embeddings, entity_all_embeddings = torch.split(
            kgat_all_embeddings, [self.n_users, self.n_entities]
        )
        return user_all_embeddings, entity_all_embeddings

    def generate_transE_score(self, hs, ts, r, user_embedding, entity_embedding):
        r"""Calculating scores for triples in KG.

        Args:
            hs (torch.Tensor): head entities
            ts (torch.Tensor): tail entities
            r (int): the relation id between hs and ts

        Returns:
            torch.Tensor: the scores of (hs, r, ts)
        """

        all_embeddings = self._get_ego_embeddings(user_embedding, entity_embedding)
        h_e = all_embeddings[hs]
        t_e = all_embeddings[ts]
        r_e = self.relation_embedding.weight[r]
        r_trans_w = self.trans_w.weight[r].view(
            self.embedding_size, self.kg_embedding_size
        )

        h_e = torch.matmul(h_e, r_trans_w)
        t_e = torch.matmul(t_e, r_trans_w)

        kg_score = torch.mul(t_e, self.tanh(h_e + r_e)).sum(dim=1)

        return kg_score
    
    def _get_kg_embedding(self, h, r, pos_t, neg_t, entity_embedding):
        h_e = entity_embedding(h).unsqueeze(1)
        pos_t_e = entity_embedding(pos_t).unsqueeze(1)
        neg_t_e = entity_embedding(neg_t).unsqueeze(1)
        r_e = self.relation_embedding(r)
        r_trans_w = self.trans_w(r).view(
            r.size(0), self.embedding_size, self.kg_embedding_size
        )

        h_e = torch.bmm(h_e, r_trans_w).squeeze(1)
        pos_t_e = torch.bmm(pos_t_e, r_trans_w).squeeze(1)
        neg_t_e = torch.bmm(neg_t_e, r_trans_w).squeeze(1)

        return h_e, r_e, pos_t_e, neg_t_e
    
    def calculate_kg_loss(self, interaction, entity_embedding):
        r"""Calculate the training loss for a batch data of KG.

        Args:
            interaction (Interaction): Interaction class of the batch.

        Returns:
            torch.Tensor: Training loss, shape: []
        """
        # get loss for training kg
        h = interaction[self.HEAD_ENTITY_ID]
        r = interaction[self.RELATION_ID]
        pos_t = interaction[self.TAIL_ENTITY_ID]
        neg_t = interaction[self.NEG_TAIL_ENTITY_ID]

        h_e, r_e, pos_t_e, neg_t_e = self._get_kg_embedding(h, r, pos_t, neg_t, entity_embedding)
        pos_tail_score = ((h_e + r_e - pos_t_e) ** 2).sum(dim=1)
        neg_tail_score = ((h_e + r_e - neg_t_e) ** 2).sum(dim=1)
        kg_loss = F.softplus(pos_tail_score - neg_tail_score).mean()
        kg_reg_loss = self.reg_loss(h_e, r_e, pos_t_e, neg_t_e)
        loss = kg_loss + self.reg_weight * kg_reg_loss

        return loss

    def update_attentive_A(self, user_embedding, entity_embedding):
        r"""Update the attention matrix using the updated embedding matrix"""

        kg_score_list, row_list, col_list = [], [], []
        # To reduce the GPU memory consumption, we calculate the scores of KG triples according to the type of relation
        for rel_idx in range(1, self.n_relations, 1):
            triple_index = torch.where(self.all_rs == rel_idx)
            kg_score = self.generate_transE_score(
                self.all_hs[triple_index], self.all_ts[triple_index], rel_idx, user_embedding, entity_embedding
            )
            row_list.append(self.all_hs[triple_index])
            col_list.append(self.all_ts[triple_index])
            kg_score_list.append(kg_score)
        kg_score = torch.cat(kg_score_list, dim=0)
        row = torch.cat(row_list, dim=0)
        col = torch.cat(col_list, dim=0)
        indices = torch.cat([row, col], dim=0).view(2, -1)
        # Current PyTorch version does not support softmax on SparseCUDA, temporarily move to CPU to calculate softmax
        A_in = torch.sparse.FloatTensor(indices, kg_score, self.matrix_size).cpu()
        A_in = torch.sparse.softmax(A_in, dim=1).to(self.device)
        self.A_in = A_in