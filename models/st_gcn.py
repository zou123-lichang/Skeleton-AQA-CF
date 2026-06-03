import torch
import torch.nn as nn
import torch.nn.functional as F
from .graph import Graph


class ConvTemporalGraphical(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, t_kernel_size=1, t_stride=1, t_padding=0, t_dilation=1,
                 bias=True):
        super().__init__()
        self.kernel_size = kernel_size
        self.conv = nn.Conv2d(in_channels, out_channels * kernel_size, kernel_size=(t_kernel_size, 1),
                              padding=(t_padding, 0), stride=(t_stride, 1), dilation=(t_dilation, 1), bias=bias)

    def forward(self, x, A):
        x = self.conv(x)
        n, kc, t, v = x.size()
        x = x.view(n, self.kernel_size, kc // self.kernel_size, t, v)
        # 核心图卷积运算：(N, K, C, T, V) * (K, V, V) -> (N, C, T, V)
        x = torch.einsum('nkctv,kvw->nctw', (x, A))
        return x.contiguous()


class st_gcn_block(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dropout=0):
        super().__init__()
        self.gcn = ConvTemporalGraphical(in_channels, out_channels, kernel_size)
        self.tcn = nn.Sequential(
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, (9, 1), (stride, 1), (4, 0)),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(dropout, inplace=True),
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, A):
        x = self.gcn(x, A)
        x = self.tcn(x)
        return self.relu(x)


class Model(nn.Module):
    def __init__(self, in_channels, num_class, graph_args,
                 edge_importance_weighting=True,
                 use_temporal_attention=False):
        super().__init__()

        self.graph = Graph(**graph_args)
        A = torch.tensor(self.graph.A, dtype=torch.float32, requires_grad=False)
        self.register_buffer('A', A)

        self.st_gcn_networks = nn.ModuleList((
            st_gcn_block(in_channels, 64, kernel_size=A.size(0), stride=1, dropout=0.2),
            st_gcn_block(64, 128, kernel_size=A.size(0), stride=2, dropout=0.3),
            st_gcn_block(128, 256, kernel_size=A.size(0), stride=2, dropout=0.3),
        ))

        if edge_importance_weighting:
            self.edge_importance = nn.ParameterList([
                nn.Parameter(torch.ones(self.A.size())) for _ in self.st_gcn_networks
            ])
        else:
            self.edge_importance = [1] * len(self.st_gcn_networks)

        # ====== A: Temporal Attention Pooling ======
        self.use_temporal_attention = use_temporal_attention
        if self.use_temporal_attention:
            # 输入: (N,256,T) -> 输出: (N,1,T)
            self.temporal_att = nn.Sequential(
                nn.Conv1d(256, 128, kernel_size=1),
                nn.ReLU(inplace=True),
                nn.Conv1d(128, 1, kernel_size=1)
            )
            self.att_mix_logit = nn.Parameter(torch.tensor(-3.0))  # sigmoid≈0.05，几乎等于baseline，保证不掉点

        self.fcn = nn.Conv2d(256, num_class, kernel_size=1)

    def forward(self, x):
        # x: (N, C, T, V)
        N, C, T, V = x.size()

        for gcn, importance in zip(self.st_gcn_networks, self.edge_importance):
            x = gcn(x, self.A * importance)  # (N,256,T',V)

        # baseline average pooling
        x_avg = F.avg_pool2d(x, x.size()[2:])  # (N,256,1,1)

        if self.use_temporal_attention:
            x_t = x.mean(dim=3)  # (N,256,T')
            att = self.temporal_att(x_t)  # (N,1,T')
            att = torch.softmax(att / 2.0, dim=2)  # 温度=2.0，注意力更平滑
            x_att = (x_t * att).sum(dim=2, keepdim=True).unsqueeze(-1)  # (N,256,1,1)

            beta = torch.sigmoid(self.att_mix_logit)
            x = (1 - beta) * x_avg + beta * x_att
        else:
            x = x_avg

        x = self.fcn(x).view(N, -1)  # (N,1)

        # 注意：不要再乘100（KIMORE标签不是0-100）
        return x
