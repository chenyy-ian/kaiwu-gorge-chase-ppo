#!/usr/bin/env python3
# -*- coding: UTF-8 -*-
###########################################################################
# Copyright © 1998 - 2026 Tencent. All Rights Reserved.
###########################################################################

"""
Author: Tencent AI Arena Authors

Feature preprocessor for treasure collection with monsters.
"""

import numpy as np
import math
import copy
from collections import deque
from agent_ppo.conf.conf import Config
from agent_ppo.feature.definition import norm, reward_process


class Preprocessor:
    """Feature preprocessor for treasure collection game with monsters."""

    def __init__(self):
        self.move_action_num = Config.ACTION_NUM
        self.reset()

    def reset(self):
        """Reset per-episode state."""
        self.step_no = 0
        self.cur_pos = (0, 0)
        self.cur_pos_norm = np.array([0.0, 0.0])
        self.last_pos_norm = np.array([0.0, 0.0])
        
        # ===== 历史位置 =====
        self.history_pos = deque(maxlen=20)
        self.position_history_5 = []
        self.position_history_10 = []
        self.move_distance_5steps = 0.0
        self.move_distance_10steps = 0.0
        
        # ===== 器官状态追踪 =====
        self.organs_states = [0] * Config.MAX_ORGANS
        self.organ_pos_map = {}
        
        # ===== 核心地图系统 =====
        self.global_visit_map = np.zeros((Config.MAP_SIZE, Config.MAP_SIZE), dtype=np.float32)
        self.global_monster_map = np.zeros((Config.MAP_SIZE, Config.MAP_SIZE), dtype=np.float32)
        self.global_danger_map = np.zeros((Config.MAP_SIZE, Config.MAP_SIZE), dtype=np.float32)
        self.global_terrain_map = np.full((Config.MAP_SIZE, Config.MAP_SIZE), -1, dtype=np.float32)
        
        # 局部地图
        self.local_visit_map = np.zeros((Config.VIEW_SIZE, Config.VIEW_SIZE), dtype=np.float32)
        self.local_monster_map = np.zeros((Config.VIEW_SIZE, Config.VIEW_SIZE), dtype=np.float32)
        self.local_danger_map = np.zeros((Config.VIEW_SIZE, Config.VIEW_SIZE), dtype=np.float32)
        self.local_terrain_map = np.zeros((Config.VIEW_SIZE, Config.VIEW_SIZE), dtype=np.float32)
        
        # ===== 位置编码 =====
        self.pos_row = [0] * Config.MAP_SIZE
        self.pos_col = [0] * Config.MAP_SIZE
        
        # ===== 怪物追踪 =====
        self.monster_count = 0
        self.monsters_info = []
        self.nearest_monster_dist = None
        self.last_nearest_monster_dist = None
        self.nearest_monster_pos = None
        self.nearest_monster_dir = None
        self.monster_speeds = []
        self.monster_accelerated = []
        self.monster_interval = None
        self.has_second_monster = False
        self.monster_was_in_view = False
        self.last_monster_positions = []
        self.monster_last_seen_step = -100
        self.is_chased = False
        
        # ===== 宝箱追踪 =====
        self.treasure_count = 0
        self.last_treasure_count = 0
        self.nearest_treasure_dist = None
        self.nearest_treasure_pos = None
        self.all_treasures = []

        # ===== 目标追踪 =====
        self.target_info = None
        self.last_target_info = None
        
        # ===== 闪现状态 =====
        self.is_flashed = True
        self.last_flash_status = 1
        self.flash_cd_remaining = 0
        self.last_flash_step = -100
        self.flash_cooldown_config = 100
        
        # ===== Buff追踪 =====
        self.buff_collected = 0
        self.last_buff_collected = 0
        self.nearest_buff_dist = None
        self.last_nearest_buff_dist = None
        self.has_active_buff = False
        
        # ===== 视野地图特征 =====
        self.treasure_flag = np.zeros(Config.VIEW_SIZE * Config.VIEW_SIZE)
        self.obstacle_flag = np.zeros(Config.VIEW_SIZE * Config.VIEW_SIZE)
        self.monster_flag = np.zeros(Config.VIEW_SIZE * Config.VIEW_SIZE)
        self.buff_flag = np.zeros(Config.VIEW_SIZE * Config.VIEW_SIZE)
        self.memory_flag = np.zeros(Config.VIEW_SIZE * Config.VIEW_SIZE)
        self.explore_flag = np.zeros(Config.VIEW_SIZE * Config.VIEW_SIZE)
        
        # ===== 特征向量 =====
        self.feature_monster_pos = np.zeros(6)
        self.feature_treasure_pos = np.zeros(6)
        self.feature_history_pos = np.zeros(6)
        
        # ===== 上一帧数据 =====
        self._obs = None
        self._extra_info = None
        self.obs = None
        self.extra_info = None
        self.last_cur_pos = None
        
        # ===== 坏动作记录 =====
        self.bad_move_ids = set()
        self.last_action = -1
        self.consecutive_stop_count = 0
        
        # ===== 徘徊检测 =====
        self.visited_count = 0
        self.circle_detected = False
        
        # ===== 地形分析缓存 =====
        self.dead_ends = set()
        self.narrow_passages = set()
        self.in_dead_end = False
        self.is_trapped_in_dead_end = False
        self.dead_end_entry_pos = None
        
        # ===== 环境配置 =====
        self.env_config = Config.ENV_CONFIG.copy()
        self.stage_boundaries = Config.get_stage_boundaries()
        
        # ===== 游戏阶段 =====
        self.current_stage = "early"
        self.stage_mult = Config.STAGE_EARLY_MULT.copy()
        self.steps_to_monster2 = self.env_config["monster_interval"]
        self.steps_to_speedup = self.env_config["monster_speedup"]

    # ==================== 位置历史更新 ====================
    def _update_position_history(self):
        """更新位置历史，计算移动距离"""
        self.position_history_5.append((self.cur_pos[0], self.cur_pos[1], self.step_no))
        self.position_history_10.append((self.cur_pos[0], self.cur_pos[1], self.step_no))
        
        if len(self.position_history_5) > 5:
            self.position_history_5.pop(0)
        if len(self.position_history_10) > 10:
            self.position_history_10.pop(0)
        
        if len(self.position_history_5) >= 2:
            start_x, start_z, _ = self.position_history_5[0]
            end_x, end_z, _ = self.position_history_5[-1]
            self.move_distance_5steps = math.sqrt((end_x - start_x)**2 + (end_z - start_z)**2)
        else:
            self.move_distance_5steps = 0.0
        
        if len(self.position_history_10) >= 2:
            start_x, start_z, _ = self.position_history_10[0]
            end_x, end_z, _ = self.position_history_10[-1]
            self.move_distance_10steps = math.sqrt((end_x - start_x)**2 + (end_z - start_z)**2)
        else:
            self.move_distance_10steps = 0.0

    # ==================== 游戏阶段管理 ====================
    def _update_game_stage(self):
        """根据当前步数更新游戏阶段"""
        step = self.step_no
        
        if step < self.stage_boundaries["early_end"]:
            self.current_stage = "early"
            self.stage_mult = Config.STAGE_EARLY_MULT.copy()
        elif step < self.stage_boundaries["mid_end"]:
            self.current_stage = "mid"
            self.stage_mult = Config.STAGE_MID_MULT.copy()
        else:
            self.current_stage = "late"
            self.stage_mult = Config.STAGE_LATE_MULT.copy()
        
        self.steps_to_monster2 = max(0, self.stage_boundaries["monster2_spawn"] - step)
        self.steps_to_speedup = max(0, self.stage_boundaries["speedup_step"] - step)

    def _get_stage_feature(self):
        """获取阶段特征"""
        if self.current_stage == "early":
            stage_code = 0.0
        elif self.current_stage == "mid":
            stage_code = 0.5
        else:
            stage_code = 1.0
        
        if self.steps_to_speedup <= 200:
            urgency = 1.0 - self.steps_to_speedup / 200.0
        else:
            urgency = 0.0
        
        return stage_code, urgency

    # ==================== 怪物相关 ====================
    def _detect_chased_status(self, monsters, hero_pos):
        """检测是否被怪物追逐"""
        if len(monsters) == 0:
            self.is_chased = False
            return False
        
        hero_x, hero_z = hero_pos
        
        for monster in monsters:
            mx, mz = monster["pos"]["x"], monster["pos"]["z"]
            dist = math.sqrt((mx - hero_x)**2 + (mz - hero_z)**2)
            
            if dist < 8:
                self.is_chased = True
                return True
        
        self.is_chased = False
        return False

    def _update_monster_memory(self, monsters):
        """更新怪物位置记忆"""
        if len(monsters) > 0:
            self.monster_last_seen_step = self.step_no
            self.last_monster_positions = []
            for monster in monsters:
                self.last_monster_positions.append({
                    "pos": (monster["pos"]["x"], monster["pos"]["z"]),
                    "step": self.step_no
                })

    def _get_opposite_direction_target(self):
        """获取怪物反方向的探索目标"""
        if len(self.last_monster_positions) == 0:
            return None
        
        hero_x, hero_z = self.cur_pos
        
        avg_mx = sum(m["pos"][0] for m in self.last_monster_positions) / len(self.last_monster_positions)
        avg_mz = sum(m["pos"][1] for m in self.last_monster_positions) / len(self.last_monster_positions)
        
        away_dx = hero_x - avg_mx
        away_dz = hero_z - avg_mz
        
        length = math.sqrt(away_dx**2 + away_dz**2) + 1e-8
        away_dx = away_dx / length * 20
        away_dz = away_dz / length * 20
        
        target_x = int(hero_x + away_dx)
        target_z = int(hero_z + away_dz)
        
        target_x = max(0, min(Config.MAP_SIZE - 1, target_x))
        target_z = max(0, min(Config.MAP_SIZE - 1, target_z))
        
        for radius in range(0, 10, 2):
            for dz in range(-radius, radius + 1):
                for dx in range(-radius, radius + 1):
                    tx, tz = target_x + dx, target_z + dz
                    if 0 <= tx < Config.MAP_SIZE and 0 <= tz < Config.MAP_SIZE:
                        if self.global_terrain_map[tz, tx] >= 1.0:
                            return {
                                "config_id": -2,
                                "distance": abs(tx - hero_x) + abs(tz - hero_z),
                                "pos": (tx, tz),
                                "type": "explore_opposite"
                            }
        return None

    # ==================== 死胡同检测 ====================
    def _detect_trapped_in_dead_end(self, monsters, hero_pos):
        """检测是否被怪物堵在死胡同"""
        hero_x, hero_z = hero_pos
        if not (0 <= hero_x < Config.MAP_SIZE and 0 <= hero_z < Config.MAP_SIZE):
            return False
        
        terrain_val = self.global_terrain_map[hero_z, hero_x]
        self.in_dead_end = (terrain_val == 4.0)
        
        if not self.in_dead_end:
            self.is_trapped_in_dead_end = False
            self.dead_end_entry_pos = None
            return False
        
        if self.dead_end_entry_pos is None:
            self.dead_end_entry_pos = (hero_x, hero_z)
        
        if len(monsters) == 0:
            self.is_trapped_in_dead_end = False
            return False
        
        for monster in monsters:
            mx, mz = monster["pos"]["x"], monster["pos"]["z"]
            dist = math.sqrt((mx - hero_x)**2 + (mz - hero_z)**2)
            
            if dist < 5:
                self.is_trapped_in_dead_end = True
                return True
        
        self.is_trapped_in_dead_end = False
        return False

    # ==================== 闪现方向检测 ====================
    def _check_flash_toward_monster(self, flash_action, monsters, hero_pos):
        """检查是否向怪物方向闪现"""
        if flash_action is None or flash_action < 8:
            return False
        
        if len(monsters) == 0:
            return False
        
        hero_x, hero_z = hero_pos
        
        flash_directions = {
            8: (1, 0), 9: (1, -1), 10: (0, -1), 11: (-1, -1),
            12: (-1, 0), 13: (-1, 1), 14: (0, 1), 15: (1, 1)
        }
        
        flash_dir = flash_directions.get(flash_action, (0, 0))
        
        for monster in monsters:
            mx, mz = monster["pos"]["x"], monster["pos"]["z"]
            monster_dir_x = mx - hero_x
            monster_dir_z = mz - hero_z
            
            dot = flash_dir[0] * monster_dir_x + flash_dir[1] * monster_dir_z
            if dot > 0:
                return True
        
        return False

    # ==================== 探索模式切换 ====================
    def _switch_to_explore_mode(self, monsters):
        """切换到探索模式（向怪物反方向探索）"""
        self.bad_move_ids.clear()
        
        if len(self.last_monster_positions) > 0:
            opposite_target = self._get_opposite_direction_target()
            if opposite_target:
                self.target_info = opposite_target
                return
        
        explore_target = self._create_explore_target()
        if explore_target:
            self.target_info = explore_target

    def _create_explore_target(self):
        """创建探索目标"""
        hero_x, hero_z = self.cur_pos
        
        best_score = -float('inf')
        best_pos = None
        
        search_radius = 40 if self.current_stage == "early" else 25
        
        for dz in range(-search_radius, search_radius + 1, 4):
            for dx in range(-search_radius, search_radius + 1, 4):
                tx, tz = hero_x + dx, hero_z + dz
                if 0 <= tx < Config.MAP_SIZE and 0 <= tz < Config.MAP_SIZE:
                    if self.global_terrain_map[tz, tx] >= 1.0:
                        visit_count = self.global_visit_map[tz, tx]
                        danger = self.global_danger_map[tz, tx]
                        distance = abs(dx) + abs(dz)
                        
                        score = -visit_count * 5.0 - danger * 2.0 - distance * 0.3
                        
                        if score > best_score:
                            best_score = score
                            best_pos = (tx, tz)
        
        if best_pos:
            return {
                "config_id": -1,
                "distance": abs(best_pos[0] - hero_x) + abs(best_pos[1] - hero_z),
                "pos": best_pos,
                "type": "explore"
            }
        return None

    # ==================== 绕圈检测 ====================
    def _detect_circle_movement(self):
        """检测绕圈行为"""
        self.circle_detected = False
        if len(self.position_history_10) >= 8:
            recent_set = set()
            for x, z, _ in self.position_history_10[-8:]:
                pos_key = (int(x), int(z))
                if pos_key in recent_set:
                    self.circle_detected = True
                    break
                recent_set.add(pos_key)
        return self.circle_detected

    # ==================== 地形分析 ====================
    def _analyze_terrain(self, map_info_21x21):
        """分析地形，识别死角和狭窄处"""
        dead_ends = set()
        narrow_passages = set()
        
        hero_x, hero_z = self.cur_pos
        
        for local_z in range(Config.VIEW_SIZE):
            for local_x in range(Config.VIEW_SIZE):
                if map_info_21x21[local_z, local_x] == 0:
                    continue
                
                global_x = hero_x + local_x - 10
                global_z = hero_z + local_z - 10
                
                if not (0 <= global_x < Config.MAP_SIZE and 0 <= global_z < Config.MAP_SIZE):
                    continue
                
                passable_count = 0
                for dz in [-1, 0, 1]:
                    for dx in [-1, 0, 1]:
                        if dx == 0 and dz == 0:
                            continue
                        nz, nx = local_z + dz, local_x + dx
                        if 0 <= nz < Config.VIEW_SIZE and 0 <= nx < Config.VIEW_SIZE:
                            if map_info_21x21[nz, nx] == 1:
                                passable_count += 1
                
                if passable_count == 1:
                    dead_ends.add((global_x, global_z))
                    self.global_terrain_map[global_z, global_x] = 4.0
                elif passable_count == 2:
                    narrow_passages.add((global_x, global_z))
                    if self.global_terrain_map[global_z, global_x] not in [2.0, 3.0]:
                        self.global_terrain_map[global_z, global_x] = 4.0
        
        return dead_ends, narrow_passages

    # ==================== 怪物加速分析 ====================
    def _analyze_monster_acceleration(self, monsters, current_step):
        """分析怪物加速状态"""
        accelerated_monsters = []
        for monster in monsters:
            speed = monster.get("speed", 1)
            if speed >= 2:
                accelerated_monsters.append(monster.get("monster_id", 0))
        return accelerated_monsters, {}

    # ==================== 危险状态下的目标选择 ====================
    def _select_target_in_danger(self, all_treasures, danger_map, hero_pos):
        """当所有可选目标都处于危险状态时，选择最优目标"""
        if not all_treasures:
            return None
        
        hero_x, hero_z = hero_pos
        
        scored_treasures = []
        for treasure in all_treasures:
            tx, tz = treasure["pos"]
            if not (0 <= tx < Config.MAP_SIZE and 0 <= tz < Config.MAP_SIZE):
                continue
            
            danger_level = danger_map[tz, tx]
            distance = treasure.get("distance", math.sqrt((tx - hero_x)**2 + (tz - hero_z)**2))
            
            total_score = 0.6 * (danger_level / 10.0) + 0.4 * (distance / 180.0)
            
            scored_treasures.append({
                "treasure": treasure,
                "score": total_score,
                "danger": danger_level,
                "distance": distance
            })
        
        if not scored_treasures:
            return None
        
        scored_treasures.sort(key=lambda x: x["score"])
        best = scored_treasures[0]
        
        return {
            "config_id": best["treasure"]["config_id"],
            "distance": best["distance"],
            "pos": best["treasure"]["pos"],
            "danger_level": best["danger"],
            "type": "treasure"
        }

    # ==================== 怪物包夹检测 ====================
    def _detect_pincer_attack(self, monsters, hero_pos):
        """检测是否被怪物包夹"""
        if len(monsters) < 2:
            return False, None, None
        
        hero_x, hero_z = hero_pos
        danger_sectors = [0.0] * 8
        
        for monster in monsters:
            mx, mz = monster["pos"]["x"], monster["pos"]["z"]
            speed = monster.get("speed", 1)
            
            dx = mx - hero_x
            dz = mz - hero_z
            
            if dx == 0 and dz == 0:
                continue
            
            angle = math.atan2(-dz, dx)
            dir_idx = int(((angle + math.pi) / (math.pi / 4)) % 8)
            
            distance = math.sqrt(dx**2 + dz**2)
            danger = (speed * 5.0) / max(1.0, distance)
            
            for offset in [-1, 0, 1]:
                idx = (dir_idx + offset) % 8
                danger_sectors[idx] += danger * (0.5 if offset != 0 else 1.0)
        
        high_danger_count = sum(1 for d in danger_sectors if d > 2.0)
        is_pincered = high_danger_count >= 5
        
        if is_pincered:
            escape_direction = min(range(8), key=lambda i: danger_sectors[i])
        else:
            escape_direction = None
        
        return is_pincered, escape_direction, danger_sectors

    # ==================== 整合多个地图信息 ====================
    def _integrate_local_maps(self):
        """将多个地图信息整合得到 local_memory_map"""
        hero_x, hero_z = self.cur_pos
        map_z = hero_z
        
        half_view = Config.VIEW_SIZE // 2
        
        top = max(0, map_z - half_view)
        bottom = min(Config.MAP_SIZE, map_z + half_view + 1)
        left = max(0, hero_x - half_view)
        right = min(Config.MAP_SIZE, hero_x + half_view + 1)
        
        dst_top = 0
        dst_bottom = bottom - top
        dst_left = 0
        dst_right = right - left
        
        if map_z - half_view < 0:
            dst_top = half_view - map_z
        if map_z + half_view + 1 > Config.MAP_SIZE:
            dst_bottom = Config.VIEW_SIZE - (map_z + half_view + 1 - Config.MAP_SIZE)
        if hero_x - half_view < 0:
            dst_left = half_view - hero_x
        if hero_x + half_view + 1 > Config.MAP_SIZE:
            dst_right = Config.VIEW_SIZE - (hero_x + half_view + 1 - Config.MAP_SIZE)
        
        self.local_visit_map.fill(0)
        self.local_monster_map.fill(0)
        self.local_danger_map.fill(0)
        self.local_terrain_map.fill(0)
        
        h, w = bottom - top, right - left
        if h > 0 and w > 0:
            self.local_visit_map[dst_top:dst_top+h, dst_left:dst_left+w] = \
                self.global_visit_map[top:bottom, left:right]
            self.local_monster_map[dst_top:dst_top+h, dst_left:dst_left+w] = \
                self.global_monster_map[top:bottom, left:right]
            self.local_danger_map[dst_top:dst_top+h, dst_left:dst_left+w] = \
                self.global_danger_map[top:bottom, left:right]
            
            terrain_slice = self.global_terrain_map[top:bottom, left:right]
            terrain_slice = np.where(terrain_slice == -1, 1.0, terrain_slice)
            self.local_terrain_map[dst_top:dst_top+h, dst_left:dst_left+w] = terrain_slice
        
        self.memory_flag = self.local_visit_map.flatten()
        
        explore = self.local_visit_map.copy()
        treasure_mask = (self.local_terrain_map == 2.0)
        explore[treasure_mask] = 2.0
        buff_mask = (self.local_terrain_map == 3.0)
        explore[buff_mask] = 3.0
        dead_end_mask = (self.local_terrain_map == 4.0)
        explore[dead_end_mask] = 4.0
        danger_mask = self.local_danger_map > 0
        explore[danger_mask] = 5.0 + self.local_danger_map[danger_mask] * 0.5
        
        self.explore_flag = explore.flatten()

    # ==================== 更新危险地图 ====================
    def _update_danger_map(self, monsters, current_step, monster_speedup_config):
        """更新危险系数地图"""
        self.global_danger_map = self.global_danger_map * 0.9
        
        for monster in monsters:
            mx, mz = monster["pos"]["x"], monster["pos"]["z"]
            speed = monster.get("speed", 1)
            
            is_accelerated = speed >= 2 or current_step >= monster_speedup_config
            danger_radius = 12 if is_accelerated else 8
            
            for dz in range(-danger_radius, danger_radius + 1):
                for dx in range(-danger_radius, danger_radius + 1):
                    nx, nz = mx + dx, mz + dz
                    if 0 <= nx < Config.MAP_SIZE and 0 <= nz < Config.MAP_SIZE:
                        dist = math.sqrt(dx**2 + dz**2)
                        if dist <= danger_radius:
                            base_danger = 10.0 * (1 - dist / danger_radius)
                            speed_mult = 1.5 if is_accelerated else 1.0
                            danger_value = base_danger * speed_mult
                            
                            self.global_danger_map[nz, nx] = min(
                                10.0,
                                self.global_danger_map[nz, nx] + danger_value
                            )

    # ==================== 核心解析函数 ====================
    def pb2struct(self, env_obs, last_action):
        """Parse environment observation into internal state."""
        if self._obs is not None:
            self.obs = copy.deepcopy(self._obs)
        if self._extra_info is not None:
            self.extra_info = copy.deepcopy(self._extra_info)
        
        self._obs = env_obs
        self._extra_info = env_obs.get("extra_info", {})
        if self._extra_info is None:
            self._extra_info = {}
        
        observation = env_obs.get("observation", {})
        frame_state = observation.get("frame_state", {})
        env_info = observation.get("env_info", {})
        
        self.step_no = observation.get("step_no", self.step_no + 1)
        heroes = frame_state.get("heroes", [])
        organs = frame_state.get("organs", [])
        monsters = frame_state.get("monsters", [])
        
        # ===== 解析英雄位置 =====
        if heroes:
            hero = heroes[0] if isinstance(heroes, list) else heroes
            self.cur_pos = (hero["pos"]["x"], hero["pos"]["z"])
        else:
            pos = env_info.get("pos", {"x": 0, "z": 0})
            self.cur_pos = (pos["x"], pos["z"])
        
        self.last_cur_pos = self.last_cur_pos or self.cur_pos
        
        # ===== 更新位置编码 =====
        self.pos_row = [0] * Config.MAP_SIZE
        self.pos_col = [0] * Config.MAP_SIZE
        if 0 <= self.cur_pos[0] < Config.MAP_SIZE:
            self.pos_row[self.cur_pos[0]] = 1
        if 0 <= self.cur_pos[1] < Config.MAP_SIZE:
            self.pos_col[self.cur_pos[1]] = 1
        
        # ===== 解析 map_info =====
        map_info = observation.get("map_info", [])
        obstacle_map = np.zeros((Config.VIEW_SIZE, Config.VIEW_SIZE), dtype=np.float32)
        passable_map = np.zeros((Config.VIEW_SIZE, Config.VIEW_SIZE), dtype=np.float32)
        
        if isinstance(map_info, list) and len(map_info) > 0:
            for r in range(min(Config.VIEW_SIZE, len(map_info))):
                row = map_info[r]
                if isinstance(row, list):
                    for c in range(min(Config.VIEW_SIZE, len(row))):
                        if row[c] == 0:
                            obstacle_map[r, c] = 1.0
                            passable_map[r, c] = 0.0
                        else:
                            passable_map[r, c] = 1.0
                        gx = self.cur_pos[0] + c - 10
                        gz = self.cur_pos[1] + r - 10
                        if 0 <= gx < Config.MAP_SIZE and 0 <= gz < Config.MAP_SIZE:
                            if self.global_terrain_map[gz, gx] == -1:
                                self.global_terrain_map[gz, gx] = 1.0 if row[c] == 1 else 0.0
        
        self.obstacle_flag = obstacle_map.flatten()
        
        # ===== 地形分析 =====
        self._analyze_terrain(passable_map)
        
        # ===== 解析怪物信息 =====
        self.last_nearest_monster_dist = self.nearest_monster_dist
        self._parse_monsters(monsters, self.cur_pos)
        
        # ===== 怪物加速分析 =====
        monster_speedup = env_info.get("monster_speed", self.env_config["monster_speedup"])
        accelerated_monsters, _ = self._analyze_monster_acceleration(monsters, self.step_no)
        self.monster_accelerated = accelerated_monsters
        
        # ===== 更新危险地图 =====
        self._update_danger_map(monsters, self.step_no, monster_speedup)
        
        # ===== 解析器官信息 =====
        self._parse_organs(organs, self.cur_pos)
        
        # ===== 解析Buff信息 =====
        self._parse_buffs(organs, self.cur_pos)
        
        # ===== 更新目标追踪 =====
        self.last_target_info = self.target_info
        
        all_in_danger = True
        for treasure in self.all_treasures:
            tx, tz = treasure["pos"]
            if 0 <= tx < Config.MAP_SIZE and 0 <= tz < Config.MAP_SIZE:
                if self.global_danger_map[tz, tx] < 3.0:
                    all_in_danger = False
                    break
        
        if all_in_danger and self.all_treasures:
            self.target_info = self._select_target_in_danger(
                self.all_treasures, self.global_danger_map, self.cur_pos
            )
        elif self.nearest_treasure_pos is not None:
            self.target_info = {
                "config_id": 0,
                "distance": self.nearest_treasure_dist or 0,
                "pos": self.nearest_treasure_pos,
                "type": "treasure"
            }
        else:
            self.target_info = None
        
        # ===== 怪物包夹检测 =====
        self._detect_pincer_attack(monsters, self.cur_pos)
        
        # ===== 更新宝箱和Buff数量 =====
        self.last_treasure_count = self.treasure_count
        self.treasure_count = env_info.get("treasures_collected", 0)
        self.last_buff_collected = self.buff_collected
        self.buff_collected = env_info.get("collected_buff", 0)
        
        # ===== 获取配置信息 =====
        self.monster_interval = env_info.get("monster_interval", self.env_config["monster_interval"])
        self.flash_cooldown_config = env_info.get("flash_cooldown", self.env_config["talent_cooldown"])
        
        # 计算闪现剩余CD
        steps_since_flash = self.step_no - self.last_flash_step
        self.flash_cd_remaining = max(0, self.flash_cooldown_config - steps_since_flash) / self.flash_cooldown_config
        self.is_flashed = self.flash_cd_remaining == 0
        
        # ===== 更新历史位置 =====
        self.history_pos.append(self.cur_pos)
        self._update_position_history()
        
        # ===== 更新位置归一化 =====
        self.last_pos_norm = self.cur_pos_norm.copy()
        self.cur_pos_norm = norm(self.cur_pos, Config.MAP_SIZE, 0)
        
        # ===== 更新访问地图 =====
        self._update_visit_map()
        
        # ===== 更新怪物地图 =====
        self._update_monster_map(monsters)
        
        # ===== 整合局部地图 =====
        self._integrate_local_maps()
        
        # ===== 更新特征向量 =====
        self.feature_monster_pos = self._get_pos_feature(
            self.nearest_monster_pos is not None, 
            self.cur_pos, 
            self.nearest_monster_pos
        )
        self.feature_treasure_pos = self._get_pos_feature(
            self.nearest_treasure_pos is not None, 
            self.cur_pos, 
            self.nearest_treasure_pos
        )
        self.feature_history_pos = self._get_pos_feature(
            len(self.history_pos) > 0, 
            self.cur_pos, 
            self.history_pos[0] if self.history_pos else None
        )
        
        # ===== 更新游戏阶段 =====
        self._update_game_stage()
        
        # ===== 更新怪物记忆 =====
        self._update_monster_memory(monsters)
        
        # ===== 检测被追逐状态 =====
        self._detect_chased_status(monsters, self.cur_pos)
        
        # ===== 检测是否被堵在死胡同 =====
        self._detect_trapped_in_dead_end(monsters, self.cur_pos)
        
        # ===== 检测绕圈 =====
        self._detect_circle_movement()
        
        # ===== 怪物消失处理 =====
        if len(monsters) == 0:
            if self.monster_was_in_view:
                self.bad_move_ids.clear()
                self.consecutive_stop_count = 0
                self.global_danger_map = self.global_danger_map * 0.5
                self._switch_to_explore_mode(monsters)
            self.monster_was_in_view = False
        else:
            self.monster_was_in_view = True
        
        self.last_action = last_action
        if last_action is not None and last_action >= 8:
            self.last_flash_step = self.step_no

    def _update_visit_map(self):
        """更新访问频率地图"""
        x, z = self.cur_pos
        if 0 <= x < Config.MAP_SIZE and 0 <= z < Config.MAP_SIZE:
            self.global_visit_map[z, x] = min(1.0, self.global_visit_map[z, x] + 0.05)
            self.visited_count = self.global_visit_map[z, x]

    def _update_monster_map(self, monsters):
        """更新怪物专属地图"""
        self.global_monster_map = self.global_monster_map * 0.9
        for monster in monsters:
            mx, mz = monster["pos"]["x"], monster["pos"]["z"]
            if 0 <= mx < Config.MAP_SIZE and 0 <= mz < Config.MAP_SIZE:
                self.global_monster_map[mz, mx] = 1.0

    def _parse_monsters(self, monsters, hero_pos):
        """解析怪物信息"""
        self.monsters_info = []
        self.monster_speeds = []
        
        monster_map = np.zeros((Config.VIEW_SIZE, Config.VIEW_SIZE), dtype=np.float32)
        
        min_dist = float('inf')
        nearest_monster = None
        nearest_dir = None
        
        hero_x, hero_z = hero_pos
        
        for monster in monsters:
            mx, mz = monster["pos"]["x"], monster["pos"]["z"]
            speed = monster.get("speed", 1)
            direction = monster.get("hero_relative_direction", 0)
            
            dist = math.sqrt((mx - hero_x)**2 + (mz - hero_z)**2)
            
            self.monsters_info.append({
                "pos": (mx, mz),
                "l2_distance": dist,
                "direction": direction,
                "speed": speed,
                "monster_id": monster.get("monster_id", 0)
            })
            self.monster_speeds.append(speed)
            
            if dist < min_dist:
                min_dist = dist
                nearest_monster = (mx, mz)
                nearest_dir = direction
            
            rel_x = mx - hero_x + 10
            rel_z = mz - hero_z + 10
            if 0 <= rel_x < Config.VIEW_SIZE and 0 <= rel_z < Config.VIEW_SIZE:
                monster_map[rel_z, rel_x] = 1.0
        
        self.monster_count = len(monsters)
        self.has_second_monster = self.monster_count >= 2
        
        self.nearest_monster_dist = min_dist if min_dist != float('inf') else None
        self.nearest_monster_pos = nearest_monster
        self.nearest_monster_dir = nearest_dir
        
        self.monster_flag = monster_map.flatten()

    def _parse_organs(self, organs, hero_pos):
        """解析器官信息"""
        treasure_map = np.zeros((Config.VIEW_SIZE, Config.VIEW_SIZE), dtype=np.float32)
        
        min_treasure_dist = float('inf')
        nearest_treasure = None
        self.all_treasures = []
        
        hero_x, hero_z = hero_pos
        
        for organ in organs:
            config_id = organ.get("config_id", 0)
            sub_type = organ.get("sub_type", 0)
            status = organ.get("status", -1)
            ox, oz = organ["pos"]["x"], organ["pos"]["z"]
            
            if config_id < len(self.organs_states):
                self.organs_states[config_id] = 1 if status != -1 else 0
            
            self.organ_pos_map[config_id] = (ox, oz)
            
            rel_x = ox - hero_x + 10
            rel_z = oz - hero_z + 10
            
            dist = math.sqrt((ox - hero_x)**2 + (oz - hero_z)**2)
            
            if sub_type == 1:  # 宝箱
                if status != -1:
                    self.all_treasures.append({
                        "pos": (ox, oz),
                        "config_id": config_id,
                        "distance": dist
                    })
                    if dist < min_treasure_dist:
                        min_treasure_dist = dist
                        nearest_treasure = (ox, oz)
                
                if 0 <= rel_x < Config.VIEW_SIZE and 0 <= rel_z < Config.VIEW_SIZE:
                    treasure_map[rel_z, rel_x] = 1.0
                    
                if 0 <= ox < Config.MAP_SIZE and 0 <= oz < Config.MAP_SIZE:
                    self.global_terrain_map[oz, ox] = 2.0
        
        self.nearest_treasure_dist = min_treasure_dist if min_treasure_dist != float('inf') else None
        self.nearest_treasure_pos = nearest_treasure
        
        self.treasure_flag = treasure_map.flatten()

    def _parse_buffs(self, organs, hero_pos):
        """解析buff信息"""
        buff_map = np.zeros((Config.VIEW_SIZE, Config.VIEW_SIZE), dtype=np.float32)
        
        min_buff_dist = float('inf')
        hero_x, hero_z = hero_pos
        
        for organ in organs:
            sub_type = organ.get("sub_type", 0)
            status = organ.get("status", -1)
            
            if sub_type == 2 and status != -1:
                bx, bz = organ["pos"]["x"], organ["pos"]["z"]
                dist = math.sqrt((bx - hero_x)**2 + (bz - hero_z)**2)
                if dist < min_buff_dist:
                    min_buff_dist = dist
                
                rel_x = bx - hero_x + 10
                rel_z = bz - hero_z + 10
                if 0 <= rel_x < Config.VIEW_SIZE and 0 <= rel_z < Config.VIEW_SIZE:
                    buff_map[rel_z, rel_x] = 1.0
                    
                if 0 <= bx < Config.MAP_SIZE and 0 <= bz < Config.MAP_SIZE:
                    self.global_terrain_map[bz, bx] = 3.0
        
        self.last_nearest_buff_dist = self.nearest_buff_dist
        self.nearest_buff_dist = min_buff_dist if min_buff_dist != float('inf') else None
        
        self.buff_flag = buff_map.flatten()

    def _get_pos_feature(self, found, cur_pos, target_pos):
        """计算位置特征向量"""
        if target_pos is None:
            return np.zeros(6)
        
        relative_pos = (target_pos[0] - cur_pos[0], target_pos[1] - cur_pos[1])
        dist = math.sqrt(relative_pos[0]**2 + relative_pos[1]**2) + 1e-4
        
        target_pos_norm = norm(target_pos, Config.MAP_SIZE, 0)
        
        dir_x = relative_pos[0] / dist
        dir_y = relative_pos[1] / dist
        
        max_dist = Config.MAP_SIZE * math.sqrt(2)
        
        return np.array([
            float(found),
            norm(dir_x, 1, -1),
            norm(dir_y, 1, -1),
            target_pos_norm[0],
            target_pos_norm[1],
            norm(dist, max_dist),
        ])

    def feature_process(self, env_obs, last_action, is_exploit=False):
        """处理特征，返回特征向量、合法动作掩码和奖励"""
        
        # 处理上一动作带来的状态更新
        if last_action is not None and last_action >= 8:
            self.last_flash_step = self.step_no
        
        pre_flash_pos = self.cur_pos if self.cur_pos != (0, 0) else None
        
        self.pb2struct(env_obs, last_action)
        
        # 检测闪现
        is_flash_action = (last_action is not None and last_action >= 8)
        flash_distance = 0
        if is_flash_action and pre_flash_pos is not None:
            dx = self.cur_pos[0] - pre_flash_pos[0]
            dz = self.cur_pos[1] - pre_flash_pos[1]
            flash_distance = math.sqrt(dx**2 + dz**2)
        
        # 检查是否向怪物方向闪现
        monsters = env_obs.get("observation", {}).get("frame_state", {}).get("monsters", [])
        flash_toward_monster = False
        if is_flash_action:
            flash_toward_monster = self._check_flash_toward_monster(last_action, monsters, pre_flash_pos)
        
        # 闪现后重置罚站计数
        if is_flash_action:
            self.consecutive_stop_count = 0
            self.bad_move_ids.clear()
        
        # 检测罚站
        if (self.last_cur_pos is not None 
            and self.cur_pos[0] == self.last_cur_pos[0] 
            and self.cur_pos[1] == self.last_cur_pos[1]):
            self.consecutive_stop_count += 1
        else:
            self.consecutive_stop_count = 0
        
        if self.consecutive_stop_count >= 3:
            self.bad_move_ids.clear()
            self.consecutive_stop_count = 0
        
        # 绕圈检测处理
        if self.circle_detected and len(monsters) == 0:
            self._switch_to_explore_mode(monsters)
        
        # 获取阶段特征
        stage_code, urgency = self._get_stage_feature()
        
        # 计算steps_to_speedup
        steps_to_speedup = max(0, self.stage_boundaries["speedup_step"] - self.step_no)
        
        # 构建视野地图特征
        view_features = np.concatenate([
            self.treasure_flag,
            self.obstacle_flag,
            self.monster_flag,
            self.buff_flag,
            self.memory_flag,
            self.explore_flag
        ])
        
        # 构建怪物额外信息特征（8维）
        monster_count_feature = np.array([
            self.monster_count,
            float(self.has_second_monster),
            np.mean(self.monster_speeds) if self.monster_speeds else 0,
            self.nearest_monster_dir if self.nearest_monster_dir is not None else 0,
            stage_code,
            urgency,
            self.flash_cd_remaining,
            float(self.is_chased),
        ])
        
        # 构建完整特征向量
        feature = np.concatenate([
            self.cur_pos_norm,
            np.array(self.pos_row),
            np.array(self.pos_col),
            np.array(self.organs_states),
            self.feature_monster_pos,
            self.feature_treasure_pos,
            self.feature_history_pos,
            monster_count_feature,
            self.get_legal_action(),
            view_features
        ]).astype(np.float32)
        
        legal_action = self.get_legal_action()

        if is_exploit:
            return feature, legal_action
        else:
            cur_explore_value = None
            if 0 <= self.cur_pos[0] < Config.MAP_SIZE and 0 <= self.cur_pos[1] < Config.MAP_SIZE:
                cur_explore_value = self.global_terrain_map[self.cur_pos[1], self.cur_pos[0]]
            
            visited_before = False
            if 0 <= self.cur_pos[0] < Config.MAP_SIZE and 0 <= self.cur_pos[1] < Config.MAP_SIZE:
                visited_before = self.global_visit_map[self.cur_pos[1], self.cur_pos[0]] > 0.1

            reward = reward_process(
                obs=self._obs,
                extra_info=self._extra_info,
                last_obs=self.obs,
                last_extra_info=self.extra_info,
                cur_pos=self.cur_pos,
                last_pos=self.last_cur_pos,
                treasure_count=self.treasure_count,
                last_treasure_count=self.last_treasure_count,
                step_no=self.step_no,
                monster_dist=self.nearest_monster_dist,
                last_monster_dist=self.last_nearest_monster_dist,
                is_flashed=self.is_flashed,
                cur_flash_status=1 if self.is_flashed else 0,
                last_flash_status=self.last_flash_status,
                wander_value=self.visited_count,
                global_map_value=self.global_visit_map[
                    self.cur_pos[1], self.cur_pos[0]
                ] if 0 <= self.cur_pos[0] < Config.MAP_SIZE and 0 <= self.cur_pos[1] < Config.MAP_SIZE else 0,
                global_explore_map=self.global_terrain_map,
                cur_pos_explore_value=cur_explore_value,
                target_info=self.target_info,
                last_target_info=self.last_target_info,
                buff_collected=self.buff_collected,
                last_buff_collected=self.last_buff_collected,
                nearest_buff_dist=self.nearest_buff_dist,
                last_nearest_buff_dist=self.last_nearest_buff_dist,
                stage_mult=self.stage_mult,
                flash_distance=flash_distance,
                is_flash_action=is_flash_action,
                in_dead_end=self.in_dead_end,
                visited_before=visited_before,
                is_trapped_in_dead_end=self.is_trapped_in_dead_end,
                flash_toward_monster=flash_toward_monster,
                steps_to_speedup=steps_to_speedup,
                is_chased=self.is_chased,
                move_distance_5steps=self.move_distance_5steps,
                move_distance_10steps=self.move_distance_10steps,
            )
            
            self.last_flash_status = 1 if self.is_flashed else 0
            
            return feature, legal_action, reward

    def get_legal_action(self):
        """返回合法动作掩码"""
        self.move_usable = True
        
        # 更新坏动作记录
        if (self.last_cur_pos is not None 
            and self.cur_pos[0] == self.last_cur_pos[0] 
            and self.cur_pos[1] == self.last_cur_pos[1]
            and self.last_action is not None
            and 0 <= self.last_action < 8):
            if self.consecutive_stop_count >= 1:
                self.bad_move_ids.add(self.last_action)
        
        legal_action = [1.0] * 16
        
        # 闪现动作检查
        if self.flash_cd_remaining > 0:
            for i in range(8, 16):
                legal_action[i] = 0.0
        
        # 禁止坏动作（但保留至少4个移动方向）
        bad_list = list(self.bad_move_ids)
        if len(bad_list) > 4:
            self.bad_move_ids = set(bad_list[-4:])
        
        for move_id in self.bad_move_ids:
            if 0 <= move_id < 8:
                legal_action[move_id] = 0.0
        
        # 确保至少4个移动方向可用
        available_moves = sum(1 for i in range(8) if legal_action[i] > 0)
        if available_moves < 4:
            self.bad_move_ids.clear()
            for i in range(8):
                legal_action[i] = 1.0
        
        return legal_action