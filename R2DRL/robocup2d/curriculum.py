from __future__ import annotations
import random
import numpy as np
from collections import deque


class CurriculumController:
    def __init__(self, init_n=1):

        self.current_n = init_n

        self.trajectories, self.traj_progress = self.load_trajectory_array("./trajectories.npz")
        self.traj_max_frame = self.traj_progress.copy()
        self.traj_max_progress_sum = float(np.sum(self.traj_max_frame))
        self.traj_curr_progress_sum = self.traj_max_progress_sum

        self.window_size = 5
        self.max_n = 11
        self.key_stats = {}

    def generate_new_key(self):
        """
        只从当前 frontier key 的 level 属于 buffer / good 的轨迹中，
        均匀随机生成新的 key。
        返回:
            key = (traj_id, frame_idx, n_control)
        如果没有可用轨迹，则返回 None
        """
        if len(self.trajectories) == 0:
            raise ValueError("no trajectories loaded")

        valid_keys = []

        valid_traj_ids = np.where(self.traj_progress > 0)[0]

        for traj_id in valid_traj_ids:
            frame_idx = int(self.traj_progress[traj_id])
            key = (int(traj_id), frame_idx, self.current_n)

            entry = self._ensure_key_stats(key)
            level = entry["level"]

            if level in ("buffer", "good"):
                valid_keys.append(key)

        if len(valid_keys) == 0:
            return None

        return random.choice(valid_keys)

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
                "level": "buffer",
            }
        return self.key_stats[key]
    
    def update_key_stats(self, key, episode_return, min_filled=5, low=-0.2, high=0.4):
        """
        更新某个 key 的统计信息，并在满足条件时推进该轨迹的课程进度。

        key:
            (traj_id, frame_idx, n_control)

        返回:
            mean_return, visits, level, advanced
        """
        traj_id, frame_idx, n_control = key
        entry = self._ensure_key_stats(key)
        # 1) 更新最近回报统计
        entry["returns"].append(float(episode_return))
        entry["visits"] += 1
        entry["mean_return"] = float(np.mean(entry["returns"]))

        # 2) 更新分类
        filled = len(entry["returns"])
        mean_return = entry["mean_return"]

        if filled < min_filled:
            entry["level"] = "buffer"
        elif mean_return < low:
            entry["level"] = "hard"
        elif mean_return > high:
            entry["level"] = "easy"
        else:
            entry["level"] = "good"

        # 3) 如果这是该轨迹当前 frontier，并且已经 easy，则推进一帧
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
                # 已经在最前面，不能再回退
                print(
                    f"[Curriculum] traj={traj_id} already at frame 0 "
                    f"(n={self.current_n})"
                )

        return entry["mean_return"], entry["visits"], entry["level"], advanced

    def advance_n(self):
        """
        直接推进控制人数 n:
        - current_n += 1
        - 所有轨迹进度重置为各自最大帧
        - 清空 key_stats
        """
        if self.current_n >= self.max_n:
            print(f"[Curriculum] current_n already at max_n={self.max_n}")
            return False
        old_n = self.current_n
        self.current_n += 1
        # 重置所有轨迹课程进度
        self.traj_progress = self.traj_max_frame.copy()
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
                vec = states[global_idx]
                state = self.decode_frame_vector(vec)
                frames.append(state)

            trajectories.append((traj_len, frames))

            # 初始课程进度 = 该轨迹最后一帧的 frame_idx
            traj_progress[traj_id] = traj_len - 1

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

    def apply_state_and_n_by_key(self, env, key):
        """
        根据 key:
        - 取出对应 traj_id / frame_idx 的状态
        - 设置 n_control
        - 把 reset 状态写进 env
        """
        traj_id, frame_idx, n_control = key

        _, state = self.get_state_by_key(traj_id, frame_idx)

        self.set_player_mask_n(env, n_control)

        env.agents.configure_reset_state(
            ball=state["ball"],
            left_players=state["left_players"],
            right_players=state["right_players"],
            body_angles=state["body_angles"],
        )

        return frame_idx, state

    def get_state_by_key(self, traj_id, frame_idx):
        """
        直接根据 traj_id 和 frame_idx 取状态
        """
        if traj_id < 0 or traj_id >= len(self.trajectories):
            raise IndexError(f"invalid traj_id={traj_id}")

        traj_len, frames = self.trajectories[traj_id]

        if frame_idx < 0 or frame_idx >= traj_len:
            raise IndexError(
                f"invalid frame_idx={frame_idx} for traj_id={traj_id}, traj_len={traj_len}"
            )

        state = frames[frame_idx]
        return frame_idx, state

    def get_frontier_stats(self):
        stats = {}

        # 1) 全局进度（百分比）
        if self.traj_max_progress_sum <= 0:
            stats["progress_ratio"] = 0.0
            stats["progress_percent"] = 0.0
        else:
            stats["progress_ratio"] = 1.0 - (
                self.traj_curr_progress_sum / self.traj_max_progress_sum
            )
            stats["progress_percent"] = 100.0 * stats["progress_ratio"]

        # 2) 当前 frontier 的 level 分布
        counts = {
            "buffer": 0,
            "good": 0,
            "hard": 0,
            "easy": 0,
        }

        total = len(self.traj_progress)

        for traj_id in range(total):
            frame_idx = int(self.traj_progress[traj_id])
            key = (int(traj_id), frame_idx, self.current_n)
            entry = self._ensure_key_stats(key)
            level = entry["level"]
            counts[level] += 1

        stats["frontier/buffer_ratio"] = counts["buffer"] / total
        stats["frontier/good_ratio"] = counts["good"] / total
        stats["frontier/hard_ratio"] = counts["hard"] / total
        stats["frontier/easy_ratio"] = counts["easy"] / total

        stats["frontier/buffer_count"] = counts["buffer"]
        stats["frontier/good_count"] = counts["good"]
        stats["frontier/hard_count"] = counts["hard"]
        stats["frontier/easy_count"] = counts["easy"]

        return stats

if __name__ == "__main__":
    controller = CurriculumController(init_n=1)

    # print("\n[Main] trajectory loading finished.")
    # print(f"[Main] number of trajectories = {len(controller.trajectories)}")

    # if len(controller.trajectories) > 0:
    #     first_len, _ = controller.trajectories[0]
    #     print(f"[Main] first trajectory length = {first_len}")