import torch
import math


class MAB(torch.nn.Module):
    def __init__(
        self, Qdim, Kdim, Vdim, number_heads,
        use_ln=False, *args, **kwargs
    ):
        super(MAB, self).__init__(*args, **kwargs)
        self.Vdim = Vdim
        self.number_heads = number_heads

        assert self.Vdim % self.number_heads == 0, \
            'the dim of features should be divisible by number_heads'

        self.Qdense = torch.nn.Linear(Qdim, self.Vdim)
        self.Kdense = torch.nn.Linear(Kdim, self.Vdim)
        self.Vdense = torch.nn.Linear(Kdim, self.Vdim)
        self.Odense = torch.nn.Linear(self.Vdim, self.Vdim)

        self.use_ln = use_ln
        if self.use_ln:
            self.ln1 = torch.nn.LayerNorm(self.Vdim)
            self.ln2 = torch.nn.LayerNorm(self.Vdim)

    def forward(self, X, Y):
        Q, K, V = self.Qdense(X), self.Kdense(Y), self.Vdense(Y)
        batch_size, dim_split = Q.shape[0], self.Vdim // self.number_heads

        Q_split = torch.cat(Q.split(dim_split, 2), 0)
        K_split = torch.cat(K.split(dim_split, 2), 0)
        V_split = torch.cat(V.split(dim_split, 2), 0)

        Attn = torch.matmul(Q_split, K_split.transpose(1, 2))
        Attn = torch.softmax(Attn / math.sqrt(dim_split), dim=-1)
        O = Q_split + torch.matmul(Attn, V_split)
        O = torch.cat(O.split(batch_size, 0), 2)

        O = O if not self.use_ln else self.ln1(O)
        O = self.Odense(O)
        O = O if not self.use_ln else self.ln2(O)

        return O


class SAB(torch.nn.Module):
    def __init__(
        self, in_dim, out_dim, number_heads,
        use_ln=False, *args, **kwargs
    ):
        super(SAB, self).__init__(*args, **kwargs)
        self.net = MAB(in_dim, in_dim, out_dim, number_heads, use_ln)

    def forward(self, X):
        return self.net(X, X)