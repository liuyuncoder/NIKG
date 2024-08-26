# -*- coding: utf-8 -*-
# @Time   : 2020/10/6
# @Author : Changxin Tian
# @Email  : cx.tian@outlook.com

r"""
KGCN
################################################

Reference:
    Hongwei Wang et al. "Knowledge graph convolution networks for recommender systems." in WWW 2019.

Reference code:
    https://github.com/hwwang55/KGCN
"""

import numpy as np
import torch
import torch.nn as nn

from recbole.model.abstract_recommender import KnowledgeRecommender
from recbole.model.init import xavier_normal_initialization
from recbole.model.loss import EmbLoss
from recbole.utils import InputType


class KGCN(KnowledgeRecommender):
    r"""KGCN is a knowledge-based recommendation model that captures inter-item relatedness effectively by mining their
    associated attributes on the KG. To automatically discover both high-order structure information and semantic
    information of the KG, we treat KG as an undirected graph and sample from the neighbors for each entity in the KG
    as their receptive field, then combine neighborhood information with bias when calculating the representation of a
    given entity.
    """
    input_type = InputType.PAIRWISE

    def __init__(self, config, dataset):
        super(KGCN, self).__init__(config, dataset)

        # load parameters info
        self.embedding_size = config["embedding_size"]
        # number of iterations when computing entity representation
        self.n_iter = config["n_iter"]
        self.aggregator_class = config["aggregator"]  # which aggregator to use
        self.reg_weight = config["reg_weight"]  # weight of l2 regularization
        self.neighbor_sample_size = config["neighbor_sample_size"]

        # define embedding
        # self.user_embedding = nn.Embedding(self.n_users, self.embedding_size)
        # self.entity_embedding = nn.Embedding(self.n_entities, self.embedding_size)
        self.relation_embedding = nn.Embedding(
            self.n_relations + 1, self.embedding_size
        )

        # define function
        self.softmax = nn.Softmax(dim=-1)
        self.linear_layers = torch.nn.ModuleList()
        for i in range(self.n_iter):
            self.linear_layers.append(
                nn.Linear(
                    self.embedding_size
                    if not self.aggregator_class == "concat"
                    else self.embedding_size * 2,
                    self.embedding_size,
                )
            )
        self.ReLU = nn.ReLU()
        self.Tanh = nn.Tanh()

        # self.bce_loss = nn.BCEWithLogitsLoss()
        # self.l2_loss = EmbLoss()

        # parameters initialization
        self.apply(xavier_normal_initialization)
        self.other_parameter_name = ["adj_entity", "adj_relation"]

    def get_neighbors(self, adj_entity, adj_relation, items):
        r"""Get neighbors and corresponding relations for each entity in items from adj_entity and adj_relation.

        Args:
            items(torch.LongTensor): The input tensor that contains item's id, shape: [batch_size, ]

        Returns:
            tuple:
                - entities(list): Entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items.
                  dimensions of entities: {[batch_size, 1],
                  [batch_size, n_neighbor],
                  [batch_size, n_neighbor^2],
                  ...,
                  [batch_size, n_neighbor^n_iter]}
                - relations(list): Relations is a list of i-iter (i = 0, 1, ..., n_iter) corresponding relations for
                  entities. Relations have the same shape as entities.
        """
        self.adj_entity = adj_entity
        self.adj_relation = adj_relation
        items = torch.unsqueeze(items, dim=1)
        entities = [items]
        relations = []
        for i in range(self.n_iter):
            index = torch.flatten(entities[i])
            neighbor_entities = torch.index_select(self.adj_entity, 0, index).reshape(
                self.batch_size, -1
            )
            neighbor_relations = torch.index_select(
                self.adj_relation, 0, index
            ).reshape(self.batch_size, -1)
            entities.append(neighbor_entities)
            relations.append(neighbor_relations)
        return entities, relations

    def mix_neighbor_vectors(
        self, neighbor_vectors, neighbor_relations, user_embeddings
    ):
        r"""Mix neighbor vectors on user-specific graph.

        Args:
            neighbor_vectors(torch.FloatTensor): The embeddings of neighbor entities(items),
                                                 shape: [batch_size, -1, neighbor_sample_size, embedding_size]
            neighbor_relations(torch.FloatTensor): The embeddings of neighbor relations,
                                                   shape: [batch_size, -1, neighbor_sample_size, embedding_size]
            user_embeddings(torch.FloatTensor): The embeddings of users, shape: [batch_size, embedding_size]

        Returns:
            neighbors_aggregated(torch.FloatTensor): The neighbors aggregated embeddings,
            shape: [batch_size, -1, embedding_size]

        """
        avg = False
        if not avg:
            user_embeddings = user_embeddings.reshape(
                self.batch_size, 1, 1, self.embedding_size
            )  # [batch_size, 1, 1, dim]
            user_relation_scores = torch.mean(
                user_embeddings * neighbor_relations, dim=-1
            )  # [batch_size, -1, n_neighbor]
            user_relation_scores_normalized = self.softmax(
                user_relation_scores
            )  # [batch_size, -1, n_neighbor]

            user_relation_scores_normalized = torch.unsqueeze(
                user_relation_scores_normalized, dim=-1
            )  # [batch_size, -1, n_neighbor, 1]
            neighbors_aggregated = torch.mean(
                user_relation_scores_normalized * neighbor_vectors, dim=2
            )  # [batch_size, -1, dim]
        else:
            neighbors_aggregated = torch.mean(
                neighbor_vectors, dim=2
            )  # [batch_size, -1, dim]
        return neighbors_aggregated

    def aggregate(self, user_embeddings, entities, relations, entity_embedding):
        r"""For each item, aggregate the entity representation and its neighborhood representation into a single vector.

        Args:
            user_embeddings(torch.FloatTensor): The embeddings of users, shape: [batch_size, embedding_size]
            entities(list): entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items.
                            dimensions of entities: {[batch_size, 1],
                            [batch_size, n_neighbor],
                            [batch_size, n_neighbor^2],
                            ...,
                            [batch_size, n_neighbor^n_iter]}
            relations(list): relations is a list of i-iter (i = 0, 1, ..., n_iter) corresponding relations for entities.
                             relations have the same shape as entities.

        Returns:
            item_embeddings(torch.FloatTensor): The embeddings of items, shape: [batch_size, embedding_size]

        """
        entity_vectors = [entity_embedding(i) for i in entities]
        relation_vectors = [self.relation_embedding(i) for i in relations]

        for i in range(self.n_iter):
            entity_vectors_next_iter = []
            for hop in range(self.n_iter - i):
                shape = (
                    self.batch_size,
                    -1,
                    self.neighbor_sample_size,
                    self.embedding_size,
                )
                self_vectors = entity_vectors[hop]
                neighbor_vectors = entity_vectors[hop + 1].reshape(shape)
                neighbor_relations = relation_vectors[hop].reshape(shape)

                neighbors_agg = self.mix_neighbor_vectors(
                    neighbor_vectors, neighbor_relations, user_embeddings
                )  # [batch_size, -1, dim]

                if self.aggregator_class == "sum":
                    output = (self_vectors + neighbors_agg).reshape(
                        -1, self.embedding_size
                    )  # [-1, dim]
                elif self.aggregator_class == "neighbor":
                    output = neighbors_agg.reshape(-1, self.embedding_size)  # [-1, dim]
                elif self.aggregator_class == "concat":
                    # [batch_size, -1, dim * 2]
                    output = torch.cat([self_vectors, neighbors_agg], dim=-1)
                    output = output.reshape(
                        -1, self.embedding_size * 2
                    )  # [-1, dim * 2]
                else:
                    raise Exception("Unknown aggregator: " + self.aggregator_class)

                output = self.linear_layers[i](output)
                # [batch_size, -1, dim]
                output = output.reshape(self.batch_size, -1, self.embedding_size)

                if i == self.n_iter - 1:
                    vector = self.Tanh(output)
                else:
                    vector = self.ReLU(output)

                entity_vectors_next_iter.append(vector)
            entity_vectors = entity_vectors_next_iter

        item_embeddings = entity_vectors[0].reshape(
            self.batch_size, self.embedding_size
        )

        return item_embeddings

    def forward(self, adj_entity, adj_relation, user_embedding, entity_embedding, user, item):
        self.batch_size = item.shape[0]
        # [batch_size, dim]
        user_e = user_embedding(user)
        # entities is a list of i-iter (i = 0, 1, ..., n_iter) neighbors for the batch of items. dimensions of entities:
        # {[batch_size, 1], [batch_size, n_neighbor], [batch_size, n_neighbor^2], ..., [batch_size, n_neighbor^n_iter]}
        entities, relations = self.get_neighbors(adj_entity, adj_relation, item)
        # [batch_size, dim]
        item_e = self.aggregate(user_e, entities, relations, entity_embedding)

        return user_e, item_e