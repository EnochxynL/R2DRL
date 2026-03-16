from __future__ import annotations
import random
import numpy as np
from collections import deque


class CurriculumController:
    def __init__(self, depth_step=1, init_n=1):

        self.depth_step = depth_step
        self.current_n = init_n
        self.trajectories = self.load_trajectory_array("trajectories.npz")

        # 回报统计窗口长度
        self.window_size = 5

        # 只有 good 池严格限制容量
        self.good_capacity = 20

        self.max_n = 11
        self.min_depth = 0
        self.max_depth = 300

        # 当前 curriculum 深度
        self.current_depth = self.max_depth
        

        # key_stats[key] = {
        #     "returns": deque(maxlen=self.window_size),
        #     "mean_return": 0.0,
        #     "visits": 0,
        # }
        self.key_stats = {}

        self.current_episode_key = None
        self.dm1_mean = 0.0

        # 只维护两层池：
        # d_pool        : 当前训练层
        # d_minus_1_pool: 候选迁移层
        self.d_pool = self._make_empty_pool(self.current_depth)
        self.d_minus_1_pool = self._make_empty_pool(
            max(self.min_depth, self.current_depth - self.depth_step)
        )
        self.dm1_returns = deque(maxlen=30)

    # ============================================================
    # pool utilities
    # ============================================================
    def _make_empty_pool(self, depth):
        return {
            "depth": depth,
            "buffer": [],
            "good": [],
            "hard": [],
            "easy": [],
        }

    def _reset_pools_for_current_depth(self):
        """
        当切换到新的 current_depth 时，重建当前 d / d-1 两层池子。
        """
        self.d_pool = self._make_empty_pool(self.current_depth)
        self.d_minus_1_pool = self._make_empty_pool(
            max(self.min_depth, self.current_depth - self.depth_step)
        )

    def _pool_levels(self):
        return ["buffer", "good", "hard", "easy"]

    def _get_pool_by_depth(self, depth):
        if depth == self.d_pool["depth"]:
            return self.d_pool
        elif depth == self.d_minus_1_pool["depth"]:
            return self.d_minus_1_pool
        else:
            return None

    def _remove_key_from_pool(self, pool, key):
        for level in self._pool_levels():
            if key in pool[level]:
                pool[level].remove(key)

    def _append_with_capacity(self, arr, key, capacity=None):
        """
        把 key 放进列表。
        - 若已有则先移除再追加到末尾
        - 如果 capacity is None，则不限制容量
        - 如果 capacity 是整数，则按 FIFO 截断
        """
        if key in arr:
            arr.remove(key)
        arr.append(key)

        if capacity is not None and len(arr) > capacity:
            arr.pop(0)

    def _pool_summary(self, pool):
        n_buffer = len(pool["buffer"])
        n_good = len(pool["good"])
        n_hard = len(pool["hard"])
        n_easy = len(pool["easy"])

        return {
            "depth": pool["depth"],
            "buffer": n_buffer,
            "good": n_good,
            "hard": n_hard,
            "easy": n_easy,
            "good_plus_easy": n_good + n_easy,
        }

    def print_pool_status(self):
        d_sum = self._pool_summary(self.d_pool)
        dm1_sum = self._pool_summary(self.d_minus_1_pool)

        print(
            f"[n={self.current_n} d={d_sum['depth']}] "
            f"buffer={d_sum['buffer']} "
            f"good={d_sum['good']} "
            f"hard={d_sum['hard']} "
            f"easy={d_sum['easy']} "
            f"good+easy={d_sum['good_plus_easy']}"
        )

        print(
            f"[n={self.current_n} d-1={dm1_sum['depth']}] "
            f"buffer={dm1_sum['buffer']} "
            f"good={dm1_sum['good']} "
            f"hard={dm1_sum['hard']} "
            f"easy={dm1_sum['easy']} "
            f"good+easy={dm1_sum['good_plus_easy']}"
        )

    # ============================================================
    # unseen key utilities
    # ============================================================
    def _collect_seen_keys(self, pool):
        """
        收集当前 pool 中已经出现过的所有 key。
        """
        seen = set()
        for level in self._pool_levels():
            seen.update(pool[level])
        return seen

    def random_unseen_key(self, depth):
        """
        返回当前 depth 下、当前 pool 中尚未出现过的 key。
        如果没有 unseen key 了，返回 None。
        """
        if len(self.trajectories) == 0:
            raise ValueError("no trajectories loaded")

        pool = self._get_pool_by_depth(depth)
        if pool is None:
            return None

        seen = self._collect_seen_keys(pool)

        candidates = []
        for traj_id in range(len(self.trajectories)):
            key = (depth, self.current_n, traj_id)
            if key not in seen:
                candidates.append(key)

        if len(candidates) == 0:
            return None

        return random.choice(candidates)

    # ============================================================
    # key sampling
    # ============================================================
    def _sample_from_existing_keys(self, choices, weights):
        """
        从已有 key 中按权重采样。
        要求 choices 非空。
        """
        if len(choices) == 0:
            return None

        total = sum(weights)
        if total <= 0:
            return random.choice(choices)

        weights = [w / total for w in weights]
        return random.choices(choices, weights=weights, k=1)[0]
    
    def sample_key_from_pool(self, pool):
        """
        自适应采样：
        - good 满了：只从 good 抽
        - good 未满时：
            * buffer 太少 -> 优先 unseen
            * buffer 足够 -> 以 buffer 为主做评估
            * good 接近满 -> 提高 good 的重采样比例
        """
        good = pool["good"]
        buffer_keys = pool["buffer"]
        hard = pool["hard"]

        g = len(good)
        b = len(buffer_keys)
        h = len(hard)
        cap = self.good_capacity

        # 1) good 已满：冻结探索，只学 good
        if g >= cap:
            if g > 0:
                return random.choice(good)
            unseen_key = self.random_unseen_key(pool["depth"])
            return unseen_key

        # 2) 计算阶段进度
        good_progress = min(1.0, g / float(cap))
        buffer_progress = min(1.0, b / float(cap))

        # 3) unseen 概率自适应
        # buffer 越少，越该探索
        # good 越接近满，越该少探索
        fresh_prob = 0.7 * (1.0 - buffer_progress) + 0.1 * (1.0 - good_progress)
        fresh_prob = max(0.05, min(0.8, fresh_prob))

        unseen_key = self.random_unseen_key(pool["depth"])

        # 如果当前还没什么已有 key，优先 unseen
        if b == 0 and g == 0 and h == 0:
            if unseen_key is not None:
                return unseen_key
            return None

        # 4) 命中探索
        if unseen_key is not None and random.random() < fresh_prob:
            return unseen_key

        # 5) 否则从已有 key 中抽，自适应分配权重
        choices = []
        weights = []

        # buffer：前中期为主力
        w_buffer = 0.55 + 0.25 * buffer_progress - 0.25 * good_progress
        w_buffer = max(0.10, w_buffer)

        # good：越接近填满越重要
        w_good = 0.10 + 0.50 * good_progress
        w_good = max(0.05, w_good)

        # hard：保留少量，避免完全遗忘
        w_hard = 0.10 + 0.10 * (1.0 - good_progress)
        w_hard = max(0.05, w_hard)

        total_existing_weight = 0.0

        if b > 0:
            choices.extend(buffer_keys)
            weights.extend([w_buffer / b] * b)
            total_existing_weight += w_buffer

        if g > 0:
            choices.extend(good)
            weights.extend([w_good / g] * g)
            total_existing_weight += w_good

        if h > 0:
            choices.extend(hard)
            weights.extend([w_hard / h] * h)
            total_existing_weight += w_hard

        if len(choices) == 0:
            return unseen_key

        return self._sample_from_existing_keys(choices, weights)

    def epsilon_generate_key(self, epsilon=0.3):
        """
        大概率从 d_pool 采样，
        小概率从 d_minus_1_pool 采样。

        d-1 的主要作用是评估是否可迁移，
        顺便也训练。
        """
        if len(self.trajectories) == 0:
            raise ValueError("no trajectories loaded")

        if self.current_depth <= self.min_depth:
            pool = self.d_pool
        else:
            use_dm1 = random.random() < epsilon
            pool = self.d_minus_1_pool if use_dm1 else self.d_pool

        key = self.sample_key_from_pool(pool)

        if key is None:
            raise RuntimeError(
                f"no available key can be sampled at n={self.current_n}, depth={pool['depth']}"
            )

        self.current_episode_key = key
        return key

    # ============================================================
    # key statistics / classification
    # ============================================================
    def _ensure_key_stats(self, key):
        if key not in self.key_stats:
            self.key_stats[key] = {
                "returns": deque(maxlen=self.window_size),
                "mean_return": 0.0,
                "visits": 0,
            }
        return self.key_stats[key]

    def update_key_return(self, key, episode_return):
        entry = self._ensure_key_stats(key)
        entry["returns"].append(float(episode_return))
        entry["visits"] += 1
        entry["mean_return"] = float(np.mean(entry["returns"]))
        depth, _, _ = key
        if depth == self.d_minus_1_pool["depth"]:
            self.dm1_returns.append(float(episode_return))
        return entry["mean_return"], entry["visits"]

    def classify_key(self, key, min_filled=5, low=-0.2, high=0.2):
        """
        分类逻辑：
        - 样本还不够 -> buffer
        - 均值太低   -> hard
        - 均值太高   -> easy
        - 中间区域   -> good

        注意：
        good 后续也可能变成 hard / easy / buffer，
        因为每次 done 后都会重新分类。
        """
        entry = self._ensure_key_stats(key)

        filled = len(entry["returns"])
        mean_return = entry["mean_return"]

        if filled < min_filled:
            return "buffer"
        elif mean_return < low:
            return "hard"
        elif mean_return > high:
            return "easy"
        else:
            return "good"

    def update_key_pool(self, key, level):
        """
        key 只会出现在一个池子的一个 level 中。
        每次重新分类时：
        1. 先从所属 pool 的所有 level 中删除
        2. 再放进新的 level

        只有 good 有容量上限，其他池不限制。
        """
        depth, _, _ = key
        pool = self._get_pool_by_depth(depth)

        if pool is None:
            # 不是当前 d 或 d-1 的 key，忽略
            return

        self._remove_key_from_pool(pool, key)

        if level == "buffer":
            self._append_with_capacity(pool["buffer"], key, capacity=None)
        elif level == "good":
            self._append_with_capacity(pool["good"], key, capacity=self.good_capacity)
        elif level == "hard":
            self._append_with_capacity(pool["hard"], key, capacity=None)
        elif level == "easy":
            self._append_with_capacity(pool["easy"], key, capacity=None)
        else:
            raise ValueError(f"unknown level: {level}")

    # ============================================================
    # curriculum advance
    # ============================================================
    def can_shift_to_d_minus_1(
        self,
        min_samples=30,
        min_mean_return=0.1,
        min_evaluated_keys=8,
    ):
        """
        推进条件：
        - d-1 层至少已有一定数量 key 被评估
        - 最近 min_samples 场属于 d-1 的 episode 加权平均回报 >= min_mean_return

        这里对正回报（如进球）乘 0.5，
        0 和负回报保持不变。
        """
        if self.current_depth <= self.min_depth:
            return False

        pool = self.d_minus_1_pool
        n_evaluated = len(pool["good"]) + len(pool["hard"]) + len(pool["easy"])

        if n_evaluated < min_evaluated_keys:
            return False

        if len(self.dm1_returns) < min_samples:
            return False

        weighted_dm1_returns = [
            0.5 * r if r > 0.0 else r
            for r in self.dm1_returns
        ]

        dm1_mean = float(np.mean(weighted_dm1_returns))
        self.dm1_mean = dm1_mean
        if dm1_mean < min_mean_return:
            return False

        print(
            f"[ShiftCheck] depth={self.current_depth} "
            f"dm1_depth={pool['depth']} "
            f"evaluated={n_evaluated} "
            f"dm1_samples={len(self.dm1_returns)} "
            f"weighted_dm1_mean={dm1_mean:.3f}"
        )
        return True

    def _advance_depth(self):
        """
        把 current_depth 往前推进一个步长：
        depth -= depth_step
        """
        old_d = self.current_depth

        new_d = max(self.min_depth, self.current_depth - self.depth_step)
        self.current_depth = new_d

        # d <- d-1
        self.d_pool = self.d_minus_1_pool
        self.d_pool["depth"] = new_d

        # 新的 d-1
        next_dm1 = max(self.min_depth, new_d - self.depth_step)
        self.d_minus_1_pool = self._make_empty_pool(next_dm1)

        self.dm1_returns.clear()

        print(f"[Curriculum] depth advanced: n={self.current_n}, {old_d} -> {new_d}")

        return self.current_depth

    def _advance_n(self):
        """
        当 depth 已经到 min_depth 后，推进 n：
        - current_n += 1
        - current_depth 重置为 max_depth
        - 重建 d / d-1 池
        """
        if self.current_n >= self.max_n:
            return self.current_n

        old_n = self.current_n
        self.current_n += 1
        self.current_depth = self.max_depth

        self._reset_pools_for_current_depth()
        self.dm1_returns.clear()

        print(
            f"[Curriculum] n advanced: {old_n} -> {self.current_n}, "
            f"depth reset -> {self.current_depth}"
        )
        return self.current_n

    def maybe_advance_curriculum(self):
        """
        先判断 d-1 是否满足推进条件。
        满足后：
        - 如果 current_depth > min_depth：推进 depth
        - 如果 current_depth == min_depth：推进 n，并把 depth 重置回 max_depth
        - 如果 n 也已经到 max_n：不再推进
        """
        if not self.can_shift_to_d_minus_1():
            return False

        # depth 还能继续往前推进
        if self.current_depth > self.min_depth:
            self._advance_depth()
            return True

        # depth 已经到底，开始推进 n
        if self.current_depth == self.min_depth and self.current_n < self.max_n:
            self._advance_n()
            return True

        # n 也到头了
        return False

    # ============================================================
    # trajectory loading
    # ============================================================
    def load_trajectory_array(self, trajectory_path: str):
        """
        输入:
            trajectory_path: trajectories.npz 路径

        返回:
            trajectories:
                [
                    (traj_len, frames),
                    ...
                ]

            frames:
                [
                    {
                        "ball": ...,
                        "left_players": ...,
                        "right_players": ...,
                        "body_angles": ...
                    },
                    ...
                ]
        """
        data = np.load(trajectory_path)

        states = data["states"]
        traj_offsets = data["traj_offsets"]
        cycles = data["cycles"]  # 调试打印用

        num_traj = len(traj_offsets) - 1
        traj_lengths = np.diff(traj_offsets)

        print(f"[Trajectory] loaded from: {trajectory_path}")
        print(f"[Trajectory] states.shape = {states.shape}")
        print(f"[Trajectory] traj_offsets.shape = {traj_offsets.shape}")
        print(f"[Trajectory] cycles.shape = {cycles.shape}")
        print(f"[Trajectory] num_traj = {num_traj}")
        print(f"[Trajectory] first 10 traj lengths = {traj_lengths[:10].tolist()}")

        trajectories = []

        for traj_id in range(num_traj):
            start = int(traj_offsets[traj_id])
            end = int(traj_offsets[traj_id + 1])

            traj_len = end - start
            frames = []

            for global_idx in range(start, end):
                vec = states[global_idx]
                state = self.decode_frame_vector(vec)
                frames.append(state)

            trajectories.append((traj_len, frames))

        if num_traj > 0:
            traj_len, frames = trajectories[0]
            print(f"[Trajectory] first traj len = {traj_len}")
            print(f"[Trajectory] first frame ball = {frames[0]['ball']}")
            print(f"[Trajectory] first frame left[0] = {frames[0]['left_players'][0]}")
            print(f"[Trajectory] first frame right[0] = {frames[0]['right_players'][0]}")

        return trajectories

    # ============================================================
    # decode frame
    # ============================================================
    def decode_frame_vector(self, vec: np.ndarray):
        vec = np.asarray(vec, dtype=np.float32)

        if vec.shape[0] != 114:
            raise ValueError(f"frame dim must be 114, got {vec.shape[0]}")

        idx = 0

        # ball: x, y, vx, vy
        ball = vec[idx:idx + 4].astype(np.float32)
        idx += 4

        # left team: 11 * (x, y, body, vx, vy)
        left_players = []
        for _ in range(11):
            x, y, body, vx, vy = vec[idx:idx + 5]
            left_players.append([x, y, body, vx, vy])
            idx += 5

        # right team: 11 * (x, y, body, vx, vy)
        right_players = []
        for _ in range(11):
            x, y, body, vx, vy = vec[idx:idx + 5]
            right_players.append([x, y, body, vx, vy])
            idx += 5

        left_players = np.array(left_players, dtype=np.float32)
        right_players = np.array(right_players, dtype=np.float32)

        body_angles = np.concatenate(
            [left_players[:, 2], right_players[:, 2]]
        ).astype(np.float32)

        return {
            "ball": ball,
            "left_players": left_players,
            "right_players": right_players,
            "body_angles": body_angles,
        }

    # ============================================================
    # mask control / reset application
    # ============================================================
    def set_player_mask_n(self, env, n):
        env.agents.set_mask_n(n)
        return env.agents.current_mask_n

    def apply_state_and_n_by_key(self, env, key):
        """
        根据 key:
        - 取出对应 depth / traj_id 的状态
        - 设置 n_control
        - 把 reset 状态写进 env
        """
        depth, n_control, traj_id = key

        frame_idx, state = self.get_state_by_key(depth, traj_id)

        self.set_player_mask_n(env, n_control)

        env.agents.configure_reset_state(
            ball=state["ball"],
            left_players=state["left_players"],
            right_players=state["right_players"],
            body_angles=state["body_angles"],
        )

        self.current_episode_key = key
        return frame_idx, state

    # ============================================================
    # depth -> state
    # ============================================================
    def get_state_by_key(self, depth, traj_id):
        """
        用 depth 映射轨迹进度。
        当前写法延续你原来的定义：
            progress = depth / 300
        """
        if traj_id < 0 or traj_id >= len(self.trajectories):
            raise IndexError(f"invalid traj_id={traj_id}")

        traj_len, frames = self.trajectories[traj_id]

        progress = max(0.0, min(1.0, depth / float(self.max_depth)))
        frame_idx = int(progress * (traj_len - 1))
        state = frames[frame_idx]

        return frame_idx, state
