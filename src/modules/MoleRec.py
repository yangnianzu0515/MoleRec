from .SetTransformer import SAB
from .gnn import GNNGraph


import torch
import math


class AdjAttenAgger(torch.nn.Module):
    def __init__(self, Qdim, Kdim, mid_dim, *args, **kwargs):
        super(AdjAttenAgger, self).__init__(*args, **kwargs)
        self.model_dim = mid_dim
        self.Qdense = torch.nn.Linear(Qdim, mid_dim)
        self.Kdense = torch.nn.Linear(Kdim, mid_dim)
        # self.use_ln = use_ln

    def forward(self, main_feat, other_feat, fix_feat, mask=None):
        Q = self.Qdense(main_feat)
        K = self.Kdense(other_feat)
        Attn = torch.matmul(Q, K.transpose(0, 1)) / math.sqrt(self.model_dim)

        if mask is not None:
            Attn = torch.masked_fill(Attn, mask, -(1 << 32))

        Attn = torch.softmax(Attn, dim=-1)
        # print(Attn[0])
        # print(mask[0])
        fix_feat = torch.diag(fix_feat)
        other_feat = torch.matmul(fix_feat, other_feat)
        O = torch.matmul(Attn, other_feat)

        return O


class MoleRecModel(torch.nn.Module):
    def __init__(
        self, global_para, substruct_para, emb_dim, voc_size,
        substruct_num, global_dim, substruct_dim, use_embedding=False,
        device=torch.device('cpu'), dropout=0.5, *args, **kwargs
    ):
        super(MoleRecModel, self).__init__(*args, **kwargs)
        self.device = device
        self.use_embedding = use_embedding

        if self.use_embedding:
            self.substruct_emb = torch.nn.Parameter(
                torch.zeros(substruct_num, emb_dim)
            )
        else:
            self.substruct_encoder = GNNGraph(**substruct_para)

        self.global_encoder = GNNGraph(**global_para)

        self.embeddings = torch.nn.ModuleList([
            torch.nn.Embedding(voc_size[0], emb_dim),
            torch.nn.Embedding(voc_size[1], emb_dim)
        ])
        self.seq_encoders = torch.nn.ModuleList([
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True),
            torch.nn.GRU(emb_dim, emb_dim, batch_first=True)
        ])
        if dropout > 0 and dropout < 1:
            self.rnn_dropout = torch.nn.Dropout(p=dropout)
        else:
            self.rnn_dropout = torch.nn.Sequential()
        self.sab = SAB(substruct_dim, substruct_dim, 2, use_ln=True)
        self.query = torch.nn.Sequential(
            torch.nn.ReLU(),
            torch.nn.Linear(emb_dim * 4, emb_dim)
        )
        self.substruct_rela = torch.nn.Linear(emb_dim, substruct_num)
        self.aggregator = AdjAttenAgger(
            global_dim, substruct_dim, max(global_dim, substruct_dim)
        )
        score_extractor = [
            torch.nn.Linear(substruct_dim, substruct_dim // 2),
            torch.nn.ReLU(),
            torch.nn.Linear(substruct_dim // 2, 1)
        ]
        self.score_extractor = torch.nn.Sequential(*score_extractor)
        self.init_weights()

    def init_weights(self):
        """Initialize weights."""
        initrange = 0.1
        for item in self.embeddings:
            item.weight.data.uniform_(-initrange, initrange)
        if self.use_embedding:
            torch.nn.init.xavier_uniform_(self.substruct_emb)

    def forward(
        self, substruct_data, mol_data, patient_data,
        ddi_mask_H, tensor_ddi_adj, average_projection
    ):
        seq1, seq2 = [], []
        for adm in patient_data:
            Idx1 = torch.LongTensor([adm[0]]).to(self.device)
            Idx2 = torch.LongTensor([adm[1]]).to(self.device)
            repr1 = self.rnn_dropout(self.embeddings[0](Idx1))
            repr2 = self.rnn_dropout(self.embeddings[1](Idx2))
            seq1.append(torch.sum(repr1, keepdim=True, dim=1))
            seq2.append(torch.sum(repr2, keepdim=True, dim=1))
        seq1 = torch.cat(seq1, dim=1)
        seq2 = torch.cat(seq2, dim=1)
        output1, hidden1 = self.seq_encoders[0](seq1)
        output2, hidden2 = self.seq_encoders[1](seq2)
        seq_repr = torch.cat([hidden1, hidden2], dim=-1)
        last_repr = torch.cat([output1[:, -1],  output2[:, -1]], dim=-1)
        patient_repr = torch.cat([seq_repr.flatten(), last_repr.flatten()])

        query = self.query(patient_repr)
        substruct_weight = torch.sigmoid(self.substruct_rela(query))

        global_embeddings = self.global_encoder(**mol_data)
        global_embeddings = torch.mm(average_projection, global_embeddings)
        substruct_embeddings = self.sab(
            self.substruct_emb.unsqueeze(0) if self.use_embedding else
            self.substruct_encoder(**substruct_data).unsqueeze(0)
        ).squeeze(0)
        molecule_embeddings = self.aggregator(
            global_embeddings, substruct_embeddings,
            substruct_weight, mask=torch.logical_not(ddi_mask_H > 0)
        )

        score = self.score_extractor(molecule_embeddings).t()

        neg_pred_prob = torch.sigmoid(score)
        neg_pred_prob = torch.matmul(neg_pred_prob.t(), neg_pred_prob)
        batch_neg = 0.0005 * neg_pred_prob.mul(tensor_ddi_adj).sum()
        return score, batch_neg
