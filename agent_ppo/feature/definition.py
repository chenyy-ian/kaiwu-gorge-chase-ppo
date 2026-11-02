#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################
"""
Author: Tencent AI Arena Authors

Data definitions and GAE computation for Pac-Man style agent.
"""

import numpy as np
import math
import copy
from common_python.utils.common_func import create_cls
from agent_ppo.conf.conf import Config


ObsData = create_cls("ObsData", feature=None, legal_action=None)
ActData = create_cls("ActData", action=None, d_action=None, prob=None, value=None)

SampleData = create_cls(
    "SampleData",
    obs=Config.DIM_OF_OBSERVATION,
    legal_action=Config.ACTION_NUM,
    act=1,
    reward=Config.VALUE_NUM,
    reward_sum=Config.VALUE_NUM,
    done=1,
    value=Config.VALUE_NUM,
    next_value=Config.VALUE_NUM,
    advantage=Config.VALUE_NUM,
    prob=Config.ACTION_NUM,
)

MONSTER_DANGER_THRESHOLD = 4


def norm(v, max_v, min_v=0):
    if isinstance(v, (tuple, list)):
        v = np.array(v)
    v = np.maximum(np.minimum(max_v, v), min_v)
    return (v - min_v) / (max_v - min_v + 1e-8)


def reward_process(obs, extra_info, last_obs, last_extra_info,
                   cur_pos, last_pos,
                   treasure_count, last_treasure_count, step_no,
                   monster_dist, last_monster_dist,
                   is_flashed, cur_flash_status, last_flash_status,
                   wander_value, global_map_value,
                   global_explore_map, cur_pos_explore_value,
                   target_info, last_target_info,
                   buff_collected=0, last_buff_collected=0,
                   nearest_buff_dist=None, last_nearest_buff_dist=None,
                   stage_mult=None, flash_distance=0, is_flash_action=False,
                   in_dead_end=False, visited_before=False,
                   is_trapped_in_dead_end=False, flash_toward_monster=False,
                   steps_to_speedup=200, is_chased=False,
                   move_distance_5steps=0, move_distance_10steps=0,
                   is_terminal=False, terminal_info=None):
    """
    修改后的奖励函数
    """
    if stage_mult is None:
        stage_mult = {"treasure": 1.0, "buff": 1.0, "explore": 1.0, "survival": 1.0}
    
    reward_dict = {}
    
    monster_in_view = monster_dist is not None and monster_dist <= 10
    monster_very_close = monster_dist is not None and monster_dist <= 3

    # ========== 1. 基础生存奖励 ==========
    survive = Config.SURVIVE_REWARD
    if monster_in_view:
        survive *= 0.5
    reward_dict["survive"] = survive * stage_mult.get("survival", 1.0)

    # ========== 2. 宝箱奖励 ==========
    treasure_reward = 0.0
    if treasure_count > last_treasure_count:
        treasure_reward = Config.TREASURE_BASE_REWARD
        
        if is_chased:
            treasure_reward *= Config.TREASURE_CHASED_BONUS
        
        if monster_very_close:
            treasure_reward *= Config.TREASURE_DANGER_BONUS
        
        treasure_reward *= stage_mult.get("treasure", 1.0)
    reward_dict["treasure"] = treasure_reward
    
    treasure_approach = 0.0
    if target_info and target_info.get("type") == "treasure":
        cur_dist = target_info.get("distance", 0)
        if last_target_info and last_target_info.get("type") == "treasure":
            last_dist = last_target_info.get("distance", cur_dist)
            delta = last_dist - cur_dist
            if delta > 0:
                treasure_approach = Config.TREASURE_APPROACH_REWARD * min(delta, 3.0)
                if is_chased:
                    treasure_approach *= 1.2
    reward_dict["treasure_approach"] = treasure_approach

    # ========== 3. Buff奖励 ==========
    buff_reward = 0.0
    if buff_collected > last_buff_collected:
        buff_reward = Config.BUFF_BASE_REWARD
        
        if 0 < steps_to_speedup <= Config.BUFF_PRE_SPEEDUP_WINDOW:
            urgency = 1.0 + (Config.BUFF_PRE_SPEEDUP_WINDOW - steps_to_speedup) / 50.0
            buff_reward *= (1.0 + urgency * Config.BUFF_PRE_SPEEDUP_BONUS)
        
        buff_reward *= stage_mult.get("buff", 1.0)
    reward_dict["buff"] = buff_reward
    
    buff_approach = 0.0
    if nearest_buff_dist is not None and last_nearest_buff_dist is not None:
        delta = last_nearest_buff_dist - nearest_buff_dist
        if delta > 0:
            buff_approach = Config.BUFF_APPROACH_REWARD * min(delta, 3.0)
            if 0 < steps_to_speedup <= Config.BUFF_PRE_SPEEDUP_WINDOW:
                buff_approach *= 1.5
    reward_dict["buff_approach"] = buff_approach

    # ========== 4. 怪物惩罚 ==========
    monster_penalty = 0.0
    if monster_dist is not None:
        if monster_dist <= MONSTER_DANGER_THRESHOLD:
            monster_penalty = Config.MONSTER_DANGER_PENALTY * (1.0 - monster_dist / MONSTER_DANGER_THRESHOLD)
        elif monster_dist < 8:
            monster_penalty = Config.MONSTER_CLOSE_PENALTY * 0.3
    reward_dict["monster"] = monster_penalty * stage_mult.get("survival", 1.0)

    # ========== 5. 闪现奖励 ==========
    flash_reward = 0.0
    if is_flash_action:
        if flash_distance >= 5:
            flash_reward = Config.FLASH_SUCCESS_REWARD
            
            if last_monster_dist is not None and last_monster_dist < 5:
                flash_reward *= Config.FLASH_ESCAPE_BONUS
            
            if is_trapped_in_dead_end:
                flash_reward *= Config.FLASH_DEAD_END_ESCAPE_BONUS
            
            if flash_toward_monster:
                flash_reward += Config.FLASH_TOWARD_MONSTER_PENALTY
                
        elif flash_distance < 2:
            flash_reward = Config.FLASH_WASTE_PENALTY
    reward_dict["flash"] = flash_reward

    # ========== 6. 闪现冷却惩罚 ==========
    flash_cd_penalty = 0.0
    if is_flashed and cur_flash_status == 0 and last_flash_status == 1:
        flash_cd_penalty = Config.FLASH_COOLDOWN_PENALTY
    reward_dict["flash_cd"] = flash_cd_penalty

    # ========== 7. 罚站惩罚 ==========
    stop_penalty = 0.0
    if last_pos and cur_pos:
        if abs(cur_pos[0] - last_pos[0]) < 0.01 and abs(cur_pos[1] - last_pos[1]) < 0.01:
            stop_penalty = Config.STOP_BASE_PENALTY
            if not monster_in_view:
                stop_penalty *= 1.5
    reward_dict["stop"] = stop_penalty

    # ========== 8. 徘徊/绕圈惩罚 ==========
    wander_penalty = 0.0
    if not monster_in_view:
        if move_distance_5steps < Config.WANDER_DISTANCE_THRESHOLD:
            wander_penalty = Config.WANDER_BASE_PENALTY
        if move_distance_10steps < Config.WANDER_DISTANCE_THRESHOLD * 1.5:
            wander_penalty += Config.WANDER_CIRCLE_PENALTY
    reward_dict["wander"] = wander_penalty

    # ========== 9. 探索奖励 ==========
    explore_reward = 0.0
    if not visited_before and not monster_very_close:
        explore_reward = Config.EXPLORE_NEW_REWARD
    elif not monster_in_view:
        if last_pos and cur_pos:
            move_dist = math.sqrt((cur_pos[0] - last_pos[0])**2 + (cur_pos[1] - last_pos[1])**2)
            if move_dist > 0.5:
                explore_reward = Config.EXPLORE_SAFE_REWARD
    reward_dict["explore"] = explore_reward * stage_mult.get("explore", 1.0)
    
    # 反方向探索额外奖励
    if target_info and target_info.get("type") == "explore_opposite":
        reward_dict["explore"] += Config.EXPLORE_OPPOSITE_DIRECTION_BONUS

    # ========== 10. 危险区域惩罚 ==========
    danger_penalty = 0.0
    if cur_pos_explore_value is not None and cur_pos_explore_value > 3.0:
        if cur_pos_explore_value >= 4.5:
            danger_penalty = -1.0
        elif cur_pos_explore_value >= 4.0:
            danger_penalty = -0.5
        else:
            danger_penalty = -0.2
    reward_dict["danger_zone"] = danger_penalty

    # ========== 11. 死胡同被困惩罚 ==========
    dead_end_penalty = 0.0
    if in_dead_end and monster_in_view:
        dead_end_penalty = Config.DEAD_END_TRAPPED_PENALTY
        if is_trapped_in_dead_end:
            dead_end_penalty *= 1.5
    reward_dict["dead_end"] = dead_end_penalty

    # ========== 12. 失败惩罚 ==========
    fail_penalty = 0.0
    if is_terminal and terminal_info:
        fail_penalty = Config.FAIL_PENALTY
    reward_dict["fail"] = fail_penalty

    total_reward = sum(reward_dict.values())
    return [total_reward]


def sample_process(list_sample_data):
    if len(list_sample_data) == 0:
        return list_sample_data
    for i in range(len(list_sample_data) - 1):
        list_sample_data[i].next_value = list_sample_data[i + 1].value
    _calc_gae(list_sample_data)
    return list_sample_data


def _calc_gae(list_sample_data):
    gae = 0.0
    gamma = Config.GAMMA
    lamda = Config.LAMDA
    for sample in reversed(list_sample_data):
        delta = -sample.value + sample.reward + gamma * sample.next_value
        gae = gae * gamma * lamda + delta
        sample.advantage = gae
        sample.reward_sum = gae + sample.value


def SampleData2NumpyData(g_data):
    return g_data.npdata


def NumpyData2SampleData(s_data):
    return SampleData(npdata=s_data)


class SampleManager:
    def __init__(self):
        self.gamma = Config.GAMMA
        self.lamda = Config.LAMDA
        self.reset()

    def reset(self):
        self.feature = []
        self.probs = []
        self.actions = []
        self.reward = []
        self.value = []
        self.adv = []
        self.tdlamret = []
        self.legal_action = []
        self.count = 0
        self.samples = []

    def add(self, feature, legal_action, prob, action, value, reward):
        self.feature.append(feature)
        self.legal_action.append(legal_action)
        self.probs.append(prob)
        self.actions.append(action)
        self.value.append(value)
        self.reward.append(reward)
        self.adv.append(np.zeros_like(value))
        self.tdlamret.append(np.zeros_like(value))
        self.count += 1

    def add_last_reward(self, reward):
        self.reward.append(reward)
        self.value.append(np.zeros_like(reward))

    def update_sample_info(self):
        last_gae = 0
        for i in range(self.count - 1, -1, -1):
            reward = self.reward[i + 1]
            next_val = self.value[i + 1]
            val = self.value[i]
            delta = reward + next_val * self.gamma - val
            last_gae = delta + self.gamma * self.lamda * last_gae
            self.adv[i] = last_gae
            self.tdlamret[i] = last_gae + val

    def sample_process(self, feature, legal_action, prob, action, value, reward):
        self.add(feature, legal_action, prob, action, value, reward)

    def process_last_frame(self, reward):
        self.add_last_reward(reward)
        self.update_sample_info()
        self.samples = self._get_game_data()

    def get_game_data(self):
        ret = self.samples
        self.samples = []
        return ret

    def _get_game_data(self):
        if self.count == 0:
            return []
        feature = np.array(self.feature).transpose()
        probs = np.array(self.probs).transpose()
        actions = np.array(self.actions).transpose()
        reward = np.array(self.reward[:-1]).transpose()
        value = np.array(self.value[:-1]).transpose()
        legal_action = np.array(self.legal_action).transpose()
        adv = np.array(self.adv).transpose()
        tdlamret = np.array(self.tdlamret).transpose()
        data = np.concatenate([feature, reward, value, tdlamret, adv, actions, probs, legal_action]).transpose()
        samples = []
        for i in range(0, self.count):
            samples.append(SampleData(npdata=data[i].astype(np.float32)))
        return samples