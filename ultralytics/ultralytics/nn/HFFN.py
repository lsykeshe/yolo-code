import torch
import torch.nn as nn


class HFFN(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(HFFN, self).__init__()
        # 定义特征金字塔网络（FPN）
        self.fpn = nn.ModuleList([
            nn.Conv2d(in_channels[i], out_channels, 1, 1, 0, bias=False) for i in range(len(in_channels))
        ])

        # 定义特征融合层
        self.conv = nn.Conv2d(out_channels * 4, out_channels, 1, 1, 0, bias=False)

    def forward(self, features):
        # 特征金字塔网络
        fpn_features = [self.fpn[i](features[i]) for i in range(len(features))]

        # 特征融合
        fused_features = torch.cat(fpn_features, dim=1)
        fused_features = self.conv(fused_features)

        return fused_features
