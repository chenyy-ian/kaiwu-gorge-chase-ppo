#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Neural network model for Gorge Chase PPO.
峡谷追猎 PPO 神经网络模型。
"""

import torch
import torch.nn as nn
import numpy as np

from agent_ppo.conf.conf import Config


def make_fc_layer(in_features, out_features):
    """Create a linear layer with orthogonal initialization."""
    fc = nn.Linear(in_features, out_features)
    nn.init.orthogonal_(fc.weight.data)
    nn.init.zeros_(fc.bias.data)
    return fc


class Model(nn.Module):
    """Hybrid model: CNN for spatial features + MLP for scalar features."""

    def __init__(self, device=None):
        super().__init__()
        self.model_name = "gorge_chase_hybrid"
        self.device = device

        # 计算标量特征维度（总特征 - 视野地图特征）
        self.view_feature_dim = 6 * Config.VIEW_SIZE * Config.VIEW_SIZE
        self.scalar_feature_dim = Config.DIM_OF_OBSERVATION - self.view_feature_dim
        
        # ===== CNN分支：处理6个21x21视野地图 =====
        # 输入: (batch, 6, 21, 21)
        self.cnn = nn.Sequential(
            nn.Conv2d(6, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 32 x 10 x 10
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 64 x 5 x 5
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * 5 * 5, 128),
            nn.ReLU(),
        )
        
        # ===== MLP分支：处理标量特征 =====
        self.scalar_mlp = nn.Sequential(
            make_fc_layer(self.scalar_feature_dim, 128),
            nn.ReLU(),
            make_fc_layer(128, 64),
            nn.ReLU(),
        )
        
        # ===== 融合层 =====
        self.fusion = nn.Sequential(
            make_fc_layer(128 + 64, 128),
            nn.ReLU(),
        )
        
        # ===== Actor-Critic 头 =====
        self.actor_head = make_fc_layer(128, Config.ACTION_NUM)
        self.critic_head = make_fc_layer(128, Config.VALUE_NUM)

    def forward(self, obs, inference=False):
        batch_size = obs.shape[0]
        
        # 分离标量特征和视野地图特征
        scalar_features = obs[:, :self.scalar_feature_dim]
        view_features = obs[:, self.scalar_feature_dim:]
        
        # 重塑视野地图特征为 (batch, 6, 21, 21)
        view_features = view_features.view(batch_size, 6, Config.VIEW_SIZE, Config.VIEW_SIZE)
        
        # CNN分支
        cnn_out = self.cnn(view_features)
        
        # MLP分支
        scalar_out = self.scalar_mlp(scalar_features)
        
        # 融合
        fused = self.fusion(torch.cat([cnn_out, scalar_out], dim=-1))
        
        # 输出
        logits = self.actor_head(fused)
        value = self.critic_head(fused)
        
        return logits, value

    def set_train_mode(self):
        self.train()

    def set_eval_mode(self):
        self.eval()