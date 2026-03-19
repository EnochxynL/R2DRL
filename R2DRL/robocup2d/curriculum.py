from __future__ import annotations
import random
import numpy as np
from collections import deque


class CurriculumController:
    def __init__(self, init_n=1, start_window_size=3, return_window_size=5):

        self.current_n = init_n

        # 取 start state 时的窗口宽度
        self.start_window_size = int(start_window_size)

        # 统计最近 return 的滑动窗口长度（只针对当前 frontier）
        self.return_window_size = int(return_window_size)

        self.trajectories, self.traj_progress = self.load_trajectory_array("./trajectories.npz")
        self.traj_max_frame = self.traj_progress.copy()
        self.traj_max_progress_sum = float(np.sum(self.traj_max_frame))
        self.traj_curr_progress_sum = self.traj_max_progress_sum

        self.max_n = 11
        self.step = 0

        self.num_traj = len(self.trajectories)

        # 当前仍可用于课程/old采样的轨迹（traj_progress > 0）
        self.active_traj_ids = set(np.where(self.traj_progress > 0)[0].astype(int).tolist())

        # ============================================================
        # 只维护“每条轨迹当前 frontier”的统计
        # ============================================================
        self.frontier_returns_by_traj = [
            deque(maxlen=self.return_window_size) for _ in range(self.num_traj)
        ]
        self.frontier_visits_by_traj = np.zeros(self.num_traj, dtype=np.int32)
        self.frontier_mean_return_by_traj = np.zeros(self.num_traj, dtype=np.float32)
        self.frontier_level_by_traj = ["buffer" for _ in range(self.num_traj)]

        # 当前 frontier 的分类计数
        self.frontier_counts = {
            "buffer": 0,
            "good": 0,
            "hard": 0,
            "easy": 0,
        }

        # 当前 frontier 按 level 分组的轨迹集合
        self.frontier_trajs_by_level = {
            "buffer": set(),
            "good": set(),
            "hard": set(),
            "easy": set(),
        }

        self._rebuild_frontier_stats()

    def generate_new_key(self):
        """
        当前 frontier 采样逻辑：
        1. 如果还有 buffer，优先只从 buffer 里抽
        2. 如果没有 buffer，再从 good / hard 里按 4:1 抽
        3. easy 不参与 new key 采样
        """
        if len(self.trajectories) == 0:
            raise ValueError("no trajectories loaded")

        if self.frontier_trajs_by_level["buffer"]:
            traj_id = min(self.frontier_trajs_by_level["buffer"])
            frame_idx = int(self.traj_progress[traj_id])
            return (traj_id, frame_idx, self.current_n)

        good_trajs = list(self.frontier_trajs_by_level["good"])
        hard_trajs = list(self.frontier_trajs_by_level["hard"])

        weighted_trajs = []
        weighted_trajs.extend(good_trajs * 4)
        weighted_trajs.extend(hard_trajs)

        if weighted_trajs:
            traj_id = int(random.choice(weighted_trajs))
            frame_idx = int(self.traj_progress[traj_id])
            return (traj_id, frame_idx, self.current_n)

        return None

    def generate_old_key(self):
        """
        先随机选一条当前进度 > 0 的轨迹，
        再从该轨迹区间 [当前进度, 起始最大帧] 中随机抽一个 frame，
        生成 old key。
        """
        if len(self.trajectories) == 0:
            raise ValueError("no trajectories loaded")

        if not self.active_traj_ids:
            return None

        traj_id = int(random.choice(tuple(self.active_traj_ids)))

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
    # frontier statistics / classification
    # ============================================================
    def _reset_frontier_stats_for_traj(self, traj_id: int):
        """
        某条轨迹出现新的 frontier 时，将其 frontier 统计重置为初始状态。
        """
        self.frontier_returns_by_traj[traj_id].clear()
        self.frontier_visits_by_traj[traj_id] = 0
        self.frontier_mean_return_by_traj[traj_id] = 0.0
        self._set_frontier_level(traj_id, "buffer")

    def _rebuild_frontier_stats(self):
        """
        全量重建当前 frontier 的分类缓存。
        只在初始化和 advance_n 后调用。
        """
        self.frontier_counts = {
            "buffer": 0,
            "good": 0,
            "hard": 0,
            "easy": 0,
        }
        self.frontier_trajs_by_level = {
            "buffer": set(),
            "good": set(),
            "hard": set(),
            "easy": set(),
        }

        for traj_id in range(self.num_traj):
            # 每次重建时，所有 frontier 统计都从初始 buffer 状态开始
            self.frontier_returns_by_traj[traj_id].clear()
            self.frontier_visits_by_traj[traj_id] = 0
            self.frontier_mean_return_by_traj[traj_id] = 0.0
            self.frontier_level_by_traj[traj_id] = "buffer"

        for traj_id in self.active_traj_ids:
            self.frontier_counts["buffer"] += 1
            self.frontier_trajs_by_level["buffer"].add(int(traj_id))

    def _set_frontier_level(self, traj_id, new_level):
        """
        增量更新某条轨迹当前 frontier 的 level。
        同时同步：
        - frontier_level_by_traj
        - frontier_counts
        - frontier_trajs_by_level
        """
        old_level = self.frontier_level_by_traj[traj_id]

        if old_level == new_level:
            return

        if traj_id in self.active_traj_ids:
            self.frontier_counts[old_level] -= 1
            self.frontier_trajs_by_level[old_level].discard(traj_id)

            self.frontier_counts[new_level] += 1
            self.frontier_trajs_by_level[new_level].add(traj_id)

        self.frontier_level_by_traj[traj_id] = new_level

    def _remove_traj_from_frontier_cache(self, traj_id):
        """
        当轨迹课程推进到 0，不再属于 active frontier 时，从缓存中移除。
        """
        old_level = self.frontier_level_by_traj[traj_id]
        self.frontier_trajs_by_level[old_level].discard(traj_id)
        self.frontier_counts[old_level] -= 1

    def _update_active_traj(self, traj_id):
        """
        根据 traj_progress[traj_id] 是否 > 0，维护 active_traj_ids。
        """
        if int(self.traj_progress[traj_id]) > 0:
            self.active_traj_ids.add(int(traj_id))
        else:
            self.active_traj_ids.discard(int(traj_id))

    def update_key_stats(self, key, episode_return, low=-0.2, high=0.4):
        """
        只更新“当前 frontier key”的统计。
        old key 不参与课程统计。

        返回:
            mean_return, visits, level, advanced
        """
        self.step += 1
        traj_id, frame_idx, n_control = key

        current_frontier_frame = int(self.traj_progress[traj_id])

        # old key：不参与课程统计，直接返回当前 frontier 的现状
        if frame_idx != current_frontier_frame:
            return (
                float(self.frontier_mean_return_by_traj[traj_id]),
                int(self.frontier_visits_by_traj[traj_id]),
                self.frontier_level_by_traj[traj_id],
                False,
            )

        # frontier key：更新 frontier 统计
        returns_buf = self.frontier_returns_by_traj[traj_id]
        returns_buf.append(float(episode_return))

        self.frontier_visits_by_traj[traj_id] += 1
        mean_return = float(np.mean(returns_buf))
        self.frontier_mean_return_by_traj[traj_id] = mean_return

        min_filled = self.return_window_size
        filled = len(returns_buf)
        all_zero = (filled >= min_filled) and all(abs(x) < 1e-8 for x in returns_buf)

        old_level = self.frontier_level_by_traj[traj_id]

        if filled < min_filled:
            new_level = "buffer"
        elif all_zero:
            new_level = "hard"
        elif mean_return < low:
            new_level = "hard"
        elif mean_return >= high:
            new_level = "easy"
        else:
            new_level = "good"

        if old_level != new_level:
            self._set_frontier_level(traj_id, new_level)

        # 如果当前 frontier 已经 easy，则推进一帧，并把新 frontier 重置为 buffer
        advanced = False

        if new_level == "easy":
            if self.traj_progress[traj_id] > 0:
                old_frame = int(self.traj_progress[traj_id])

                self.traj_progress[traj_id] -= 1
                self.traj_curr_progress_sum -= 1.0

                new_frame = int(self.traj_progress[traj_id])
                advanced = True

                # 更新 active 集合
                self._update_active_traj(traj_id)

                if new_frame > 0:
                    # 新 frontier 是新的课程点，统计全部重置
                    self._reset_frontier_stats_for_traj(traj_id)
                else:
                    # 已不再属于 active frontier，移出统计缓存
                    self._remove_traj_from_frontier_cache(traj_id)
                    self.frontier_returns_by_traj[traj_id].clear()
                    self.frontier_visits_by_traj[traj_id] = 0
                    self.frontier_mean_return_by_traj[traj_id] = 0.0
                    self.frontier_level_by_traj[traj_id] = "buffer"

                print(
                    f"[Curriculum] traj={traj_id} advance: "
                    f"{old_frame} -> {new_frame} (n={self.current_n})"
                )

        return (
            float(self.frontier_mean_return_by_traj[traj_id]),
            int(self.frontier_visits_by_traj[traj_id]),
            self.frontier_level_by_traj[traj_id],
            advanced,
        )

    def advance_n(self):
        if self.current_n >= self.max_n:
            print(f"[Curriculum] current_n already at max_n={self.max_n}")
            return False

        old_n = self.current_n
        self.current_n += 1

        # 重置所有轨迹课程进度
        self.traj_progress = self.traj_max_frame.copy()
        self.traj_curr_progress_sum = self.traj_max_progress_sum

        # 重建 active 轨迹集合
        self.active_traj_ids = set(np.where(self.traj_progress > 0)[0].astype(int).tolist())

        # 所有 frontier 作为新课程点重新开始
        self._rebuild_frontier_stats()

        print(
            f"[Curriculum] n advanced: {old_n} -> {self.current_n}, "
            f"all traj_progress reset"
        )
        return True

    # ============================================================
    # trajectory loading
    # ============================================================
    def load_trajectory_array(self, trajectory_path: str):
        data = np.load(trajectory_path)

        starts = data["states"]
        traj_offsets = data["traj_offsets"]
        cycles = data["cycles"]

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
                frame = self.decode_frame_vector(vec)
                frames.append(frame)

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

        ball = vec[idx:idx + 4].astype(np.float32)
        idx += 4

        left_players = []
        for _ in range(11):
            x, y, body, vx, vy = vec[idx:idx + 5]
            left_players.append([x, y, body, vx, vy])
            idx += 5

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
        - key 仍然是 (traj_id, frame_idx, n_control)
        - 当 start_window_size == 1 时，直接取 frame_idx
        - 当 start_window_size > 1 时，把 frame_idx 看作窗口右端点，
          从 [frame_idx - start_window_size + 1, frame_idx] 中随机采样实际 frame_id
        """
        traj_id, frame_idx, n_control = key

        if traj_id < 0 or traj_id >= len(self.trajectories):
            raise IndexError(f"invalid traj_id={traj_id}")

        traj_len, frames = self.trajectories[traj_id]

        if frame_idx < 0 or frame_idx >= traj_len:
            raise IndexError(
                f"invalid frame_idx={frame_idx} for traj_id={traj_id}, traj_len={traj_len}"
            )

        start_idx = max(0, int(frame_idx) - self.start_window_size + 1)
        end_idx = int(frame_idx)

        sampled_frame_idx = random.randint(start_idx, end_idx)
        start = frames[sampled_frame_idx]

        return sampled_frame_idx, start

    def get_frontier_stats(self):
        """
        O(1) 返回当前 frontier 统计。
        """
        stats = {}

        if self.traj_max_progress_sum <= 0:
            stats["progress_ratio"] = 0.0
            stats["progress_percent"] = 0.0
        else:
            stats["progress_ratio"] = (
                self.traj_curr_progress_sum / self.traj_max_progress_sum
            )
            stats["progress_percent"] = 100.0 * stats["progress_ratio"]

        stats["frontier/buffer_count"] = self.frontier_counts["buffer"]
        stats["frontier/good_count"] = self.frontier_counts["good"]
        stats["frontier/hard_count"] = self.frontier_counts["hard"]
        stats["frontier/easy_count"] = self.frontier_counts["easy"]

        return stats


if __name__ == "__main__":
    controller = CurriculumController(init_n=1)