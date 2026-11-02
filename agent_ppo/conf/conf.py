#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Configuration for Pac-Man style agent (dodge monsters, collect treasures).
吃豆人风格智能体配置（躲避怪物追击，收集宝箱）。
"""


class Config:

    # Discount factor for RL / RL中的回报折扣
    GAMMA = 0.99

    # GAE lambda parameter / GAE lambda参数
    LAMDA = 0.95

    # Initial learning rate / 初始学习率
    INIT_LEARNING_RATE_START = 0.0003

    # Entropy regularization coefficient / 熵正则化系数
    BETA_START = 0.0005

    # PPO clip parameter / PPO裁剪参数
    CLIP_PARAM = 0.2

    # Value function loss coefficient / 价值函数损失系数
    VF_COEF = 1.0

    # Gradient clipping range / 梯度裁剪范围
    GRAD_CLIP_RANGE = 0.5

    # Action space: 8 directional moves + 8 flash moves = 16 actions
    ACTION_NUM = 8 + 8

    # ========== 环境配置参数 ==========
    ENV_CONFIG = {
        "treasure_count": 10,
        "buff_count": 2,
        "buff_cooldown": 100,
        "talent_cooldown": 100,
        "monster_interval": 500,
        "monster_speedup": 700,
        "max_step": 1000,
    }
    
    @classmethod
    def get_stage_boundaries(cls):
        return {
            "early_end": int(min(300, cls.ENV_CONFIG["monster_interval"] * 0.6)),
            "mid_end": cls.ENV_CONFIG["monster_speedup"],
            "monster2_spawn": cls.ENV_CONFIG["monster_interval"],
            "speedup_step": cls.ENV_CONFIG["monster_speedup"],
        }

    FEATURES = [
        2,              # 当前位置归一化
        128,            # 位置独热行编码
        128,            # 位置独热列编码
        15,             # 器官状态
        6,              # 最近怪物位置特征
        6,              # 最近宝箱位置特征
        6,              # 历史位置特征
        8,              # 怪物信息（扩展为8维）
        16,             # 合法动作掩码
        6 * 21 * 21,   # 视野地图特征
    ]

    FEATURE_SPLIT_SHAPE = FEATURES
    FEATURE_LEN = sum(FEATURE_SPLIT_SHAPE)
    DIM_OF_OBSERVATION = FEATURE_LEN

    VALUE_NUM = 1

    DATA_SPLIT_SHAPE = [
        FEATURE_LEN, VALUE_NUM, VALUE_NUM, VALUE_NUM, VALUE_NUM,
        1, VALUE_NUM, VALUE_NUM, ACTION_NUM, ACTION_NUM,
    ]
    data_len = sum(DATA_SPLIT_SHAPE)
    SAMPLE_DIM = data_len

    MAP_SIZE = 128
    VIEW_SIZE = 21
    MAX_ORGANS = 15
    MAX_STEP = 1000

    # ========== 基础奖励权重 ==========
    SURVIVE_REWARD = 0.1
    
    # ========== 宝箱奖励 ==========
    TREASURE_BASE_REWARD = 25.0        # 基础收集奖励
    TREASURE_CHASED_BONUS = 1.3        # 被怪物追时倍率
    TREASURE_DANGER_BONUS = 1.5        # 危险中收集额外奖励
    TREASURE_APPROACH_REWARD = 0.3     # 靠近宝箱奖励
    
    # ========== Buff奖励 ==========
    BUFF_BASE_REWARD = 15.0            # 基础拾取奖励
    BUFF_EARLY_BONUS = 1.5             # 前期倍率
    BUFF_PRE_SPEEDUP_BONUS = 2.5       # 加速前价值倍率
    BUFF_APPROACH_REWARD = 0.2         # 靠近buff奖励
    BUFF_PRE_SPEEDUP_WINDOW = 100      # 加速前窗口期
    
    # ========== 怪物惩罚 ==========
    MONSTER_CLOSE_PENALTY = -0.3
    MONSTER_DANGER_PENALTY = -0.5
    
    # ========== 闪现奖励 ==========
    FLASH_SUCCESS_REWARD = 0.5
    FLASH_ESCAPE_BONUS = 1.5
    FLASH_DEAD_END_ESCAPE_BONUS = 2.0
    FLASH_WASTE_PENALTY = -0.5
    FLASH_TOWARD_MONSTER_PENALTY = -1.0
    FLASH_COOLDOWN_PENALTY = -0.3
    
    # ========== 罚站和徘徊惩罚 ==========
    STOP_BASE_PENALTY = -0.3
    STOP_CONSECUTIVE_PENALTY = -0.5
    WANDER_BASE_PENALTY = -0.1
    WANDER_CIRCLE_PENALTY = -0.3
    WANDER_DISTANCE_THRESHOLD = 3.0
    
    # ========== 探索奖励 ==========
    EXPLORE_NEW_REWARD = 0.15
    EXPLORE_SAFE_REWARD = 0.05
    EXPLORE_OPPOSITE_DIRECTION_BONUS = 0.2
    
    # ========== 失败惩罚 ==========
    FAIL_PENALTY = -20.0
    DEAD_END_TRAPPED_PENALTY = -0.3
    
    # ========== 阶段倍率 ==========
    STAGE_EARLY_MULT = {
        "treasure": 1.2,
        "buff": 1.5,
        "explore": 1.5,
        "survival": 0.5,
    }
    
    STAGE_MID_MULT = {
        "treasure": 1.3,
        "buff": 1.2,
        "explore": 1.0,
        "survival": 1.2,
    }
    
    STAGE_LATE_MULT = {
        "treasure": 1.0,
        "buff": 0.5,
        "explore": 0.3,
        "survival": 2.0,
    }