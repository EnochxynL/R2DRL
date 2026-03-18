from __future__ import annotations
import random
import numpy as np
from collections import deque


class CurriculumController:
    def __init__(self, init_n=1,window_size=1):

        self.current_n = init_n
        self.window_size = int(window_size)
        self.trajectories, self.traj_progress = self.load_trajectory_array("./trajectories.npz")
        self.traj_max_frame = self.traj_progress.copy()
        self.traj_max_progress_sum = float(np.sum(self.traj_max_frame))
        self.traj_curr_progress_sum = self.traj_max_progress_sum

        self.window_size = 5
        self.max_n = 11
        self.key_stats = {}
        self.step = 0

        # 历史池（所有已创建 key）的全局分类计数
        self.hist_buffer_count = 0
        self.hist_good_count = 0
        self.hist_hard_count = 0
        self.hist_easy_count = 0

    def generate_new_key(self):
        """
        当前 frontier 采样逻辑：
        1. 如果还有 buffer，优先只从 buffer 里抽
        2. 如果没有 buffer，再从 good / hard 里按 4:1 抽
        3. easy 不参与 new key 采样
        """
        if len(self.trajectories) == 0:
            raise ValueError("no trajectories loaded")

        buffer_keys = []
        good_keys = []
        hard_keys = []

        valid_traj_ids = np.where(self.traj_progress > 0)[0]

        for traj_id in valid_traj_ids:
            frame_idx = int(self.traj_progress[traj_id])
            key = (int(traj_id), frame_idx, self.current_n)

            entry = self._ensure_key_stats(key)
            level = entry["level"]


            if level == "buffer":
                buffer_keys.append(key)
            elif level == "good":
                good_keys.append(key)
            elif level == "hard":
                hard_keys.append(key)
            # easy 不加入

        # 1) 只要还有 buffer，就只从 buffer 抽
        if len(buffer_keys) > 0:
            buffer_keys.sort(key=lambda x: x[0])  # x[0] = traj_id
            return buffer_keys[0]

        # 2) 没有 buffer 时，从 good / hard 按 4:1 抽
        weighted_keys = []
        weighted_keys.extend(good_keys * 4)
        weighted_keys.extend(hard_keys * 1)

        if len(weighted_keys) > 0:
            return random.choice(weighted_keys)

        # 3) 都没有则返回 None
        return None

    def generate_old_key(self):
        """
        先随机选一条当前进度 > 0 的轨迹，
        再从该轨迹区间 [当前进度, 起始最大帧] 中随机抽一个 frame，
        生成 old key。

        返回:
            key = (traj_id, frame_idx, n_control)

        如果没有可用轨迹，则返回 None
        """
        if len(self.trajectories) == 0:
            raise ValueError("no trajectories loaded")

        valid_traj_ids = np.where(self.traj_progress > 0)[0]

        if len(valid_traj_ids) == 0:
            return None

        traj_id = int(random.choice(valid_traj_ids))

        curr_frame = int(self.traj_progress[traj_id])
        max_frame = int(self.traj_max_frame[traj_id])

        frame_idx = random.randint(curr_frame, max_frame)

        return (traj_id, frame_idx, self.current_n)

    def generate_key(self, p_new=0.2):
        """
        按给定概率混合采样:
        - p_new 概率采样 new key
        - 1 - p_new 概率采样 old key
        """
        use_new = (random.random() < p_new)

        if use_new:
            key = self.generate_new_key()
            if key is not None:
                return key
            return self.generate_old_key()
        else:
            key = self.generate_old_key()
            if key is not None:
                return key
            return self.generate_new_key()  
    # ============================================================
    # key statistics / classification
    # ============================================================
    def _ensure_key_stats(self, key):
        if key not in self.key_stats:
            self.key_stats[key] = {
                "returns": deque(maxlen=self.window_size),
                "mean_return": 0.0,
                "visits": 0,
                "all_returns": [],
                "level": "buffer",
            }
            self.hist_buffer_count += 1   # 新 key 进入历史池，初始是 buffer
        return self.key_stats[key]
    
    def update_key_stats(self, key, episode_return, min_filled=5, low=-0.2, high=0.4):
        """
        更新某个 key 的统计信息，并在满足条件时推进该轨迹的课程进度。

        key:
            (traj_id, frame_idx, n_control)

        返回:
            mean_return, visits, level, advanced
        """
        self.step += 1
        traj_id, frame_idx, n_control = key
        entry = self._ensure_key_stats(key)

        old_level = entry["level"]

        # 1) 更新最近回报统计
        r = float(episode_return)
        entry["returns"].append(r)
        entry["visits"] += 1
        entry["all_returns"].append(r)
        entry["mean_return"] = float(np.mean(entry["returns"]))

        # 2) 更新分类
        filled = len(entry["returns"])
        mean_return = entry["mean_return"]

        all_zero = (filled >= min_filled) and all(abs(x) < 1e-8 for x in entry["returns"])

        if filled < min_filled:
            entry["level"] = "buffer"
        elif all_zero:
            entry["level"] = "hard"
        elif mean_return < low:
            entry["level"] = "hard"
        elif mean_return >= high:
            entry["level"] = "easy"
        else:
            entry["level"] = "good"

        new_level = entry["level"]

        # 3) 历史池分类计数更新
        if new_level != old_level:
            self._dec_hist_level_count(old_level)
            self._inc_hist_level_count(new_level)

        # 4) 如果这是该轨迹当前 frontier，并且已经 easy，则推进一帧
        advanced = False

        if frame_idx == int(self.traj_progress[traj_id]) and entry["level"] == "easy":
            if self.traj_progress[traj_id] > 0:
                old_frame = int(self.traj_progress[traj_id])

                self.traj_progress[traj_id] -= 1
                self.traj_curr_progress_sum -= 1.0

                new_frame = int(self.traj_progress[traj_id])
                advanced = True

                print(
                    f"[Curriculum] traj={traj_id} advance: "
                    f"{old_frame} -> {new_frame} (n={self.current_n})"
                )
            else:
                print(
                    f"[Curriculum] traj={traj_id} already at frame 0 "
                    f"(n={self.current_n})"
                )

        return entry["mean_return"], entry["visits"], entry["level"], advanced

    def advance_n(self):
        if self.current_n >= self.max_n:
            print(f"[Curriculum] current_n already at max_n={self.max_n}")
            return False

        old_n = self.current_n
        self.current_n += 1

        # 重置所有轨迹课程进度
        self.traj_progress = self.traj_max_frame.copy()
        self.traj_curr_progress_sum = self.traj_max_progress_sum

        # 清空旧统计（因为 key 里包含 n_control）
        self.key_stats.clear()

        print(
            f"[Curriculum] n advanced: {old_n} -> {self.current_n}, "
            f"all traj_progress reset"
        )
        return True


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

            traj_progress:
                shape = (num_traj,)
                每条轨迹当前进度的初始值，按顺序保存为 traj_len - 1
        """
        data = np.load(trajectory_path)

        starts = data["states"]
        traj_offsets = data["traj_offsets"]
        cycles = data["cycles"]  # 调试打印用

        num_traj = len(traj_offsets) - 1
        traj_lengths = np.diff(traj_offsets)

        print(f"[Trajectory] loaded from: {trajectory_path}")
        print(f"[Trajectory] starts.shape = {starts.shape}")
        print(f"[Trajectory] traj_offsets.shape = {traj_offsets.shape}")
        print(f"[Trajectory] cycles.shape = {cycles.shape}")

        print(f"[Trajectory] num_traj = {num_traj}")
        print(f"[Trajectory] traj_lengths = {traj_lengths.tolist()}")
        print(f"[Trajectory] min_len = {traj_lengths.min()}")
        print(f"[Trajectory] max_len = {traj_lengths.max()}")
        print(f"[Trajectory] mean_len = {traj_lengths.mean():.2f}")
        print(f"[Trajectory] total_frames = {traj_lengths.sum()}")

        trajectories = []
        traj_progress = np.zeros(num_traj, dtype=np.int32)

        for traj_id in range(num_traj):
            start = int(traj_offsets[traj_id])
            end = int(traj_offsets[traj_id + 1])

            traj_len = end - start
            frames = []

            for global_idx in range(start, end):
                vec = starts[global_idx]
                start = self.decode_frame_vector(vec)
                frames.append(start)

            trajectories.append((traj_len, frames))

            # 初始课程进度 = 该轨迹最后倒数2帧的 frame_idx
            traj_progress[traj_id] = traj_len - 3

        print(f"[Curriculum] traj_progress.shape = {traj_progress.shape}")
        print(f"[Curriculum] first 10 traj_progress = {traj_progress[:10].tolist()}")

        return trajectories, traj_progress

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

    def apply_start_and_n_by_key(self, env, key):
        """
        根据 key:
        - 取出对应 traj_id / frame_idx 的状态
        - 设置 n_control
        - 把 reset 状态写进 env
        """
        traj_id, frame_idx, n_control = key

        _, start = self.get_starts_by_key(key)

        self.set_player_mask_n(env, n_control)

        env.agents.configure_reset_start(
            ball=start["ball"],
            left_players=start["left_players"],
            right_players=start["right_players"],
            body_angles=start["body_angles"],
        )

        return frame_idx, start

    def get_starts_by_key(self, key):
        """
        根据 key 取状态。

        语义：
        - key 仍然是 (traj_id, frame_idx, n_control)
        - 当 window_size == 1 时，直接取 frame_idx
        - 当 window_size > 1 时，把 frame_idx 看作窗口右端点，
        从 [frame_idx - window_size + 1, frame_idx] 中随机采样实际 frame_id
        """
        traj_id, frame_idx, n_control = key

        if traj_id < 0 or traj_id >= len(self.trajectories):
            raise IndexError(f"invalid traj_id={traj_id}")

        traj_len, frames = self.trajectories[traj_id]

        if frame_idx < 0 or frame_idx >= traj_len:
            raise IndexError(
                f"invalid frame_idx={frame_idx} for traj_id={traj_id}, traj_len={traj_len}"
            )

        # window_size=1 时退化为原来的单帧逻辑
        start_idx = max(0, int(frame_idx) - self.window_size + 1)
        end_idx = int(frame_idx)

        sampled_frame_idx = random.randint(start_idx, end_idx)
        start = frames[sampled_frame_idx]

        return sampled_frame_idx, start
    
    def get_frontier_stats(self):
        stats = {}

        if self.traj_max_progress_sum <= 0:
            stats["progress_ratio"] = 0.0
            stats["progress_percent"] = 0.0
        else:
            stats["progress_ratio"] = (
                self.traj_curr_progress_sum / self.traj_max_progress_sum
            )
            stats["progress_percent"] = 100.0 * stats["progress_ratio"]

        stats["hist/buffer_count"] = self.hist_buffer_count
        stats["hist/good_count"] = self.hist_good_count
        stats["hist/hard_count"] = self.hist_hard_count
        stats["hist/easy_count"] = self.hist_easy_count

        return stats

    def _inc_hist_level_count(self, level):
        if level == "buffer":
            self.hist_buffer_count += 1
        elif level == "good":
            self.hist_good_count += 1
        elif level == "hard":
            self.hist_hard_count += 1
        elif level == "easy":
            self.hist_easy_count += 1
        else:
            raise ValueError(f"unknown level: {level}")

    def _dec_hist_level_count(self, level):
        if level == "buffer":
            self.hist_buffer_count -= 1
        elif level == "good":
            self.hist_good_count -= 1
        elif level == "hard":
            self.hist_hard_count -= 1
        elif level == "easy":
            self.hist_easy_count -= 1
        else:
            raise ValueError(f"unknown level: {level}")
        
if __name__ == "__main__":
    controller = CurriculumController(init_n=1)

    # print("\n[Main] trajectory loading finished.")
    # print(f"[Main] number of trajectories = {len(controller.trajectories)}")

    # if len(controller.trajectories) > 0:
    #     first_len, _ = controller.trajectories[0]
    #     print(f"[Main] first trajectory length = {first_len}")