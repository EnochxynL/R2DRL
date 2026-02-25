from __future__ import annotations

import torch
import os
import numpy as np
from .protocols import P
from . import ipc
from .logging_utils import get_env_logger
from .config import load_env_args, EnvConfig
from .runtime import Runtime
from .agents import Agents


class Robocup2dEnv:

    def __init__(self, cfg="robocup.yaml", **env_args):

        self.log = get_env_logger("robocup_env")
        self.config = EnvConfig(load_env_args(cfg, env_args))

        self.agent_mask = np.zeros(self.config.n1, dtype=bool)

        self.child_env = os.environ.copy()
        base_ld = os.environ.get("LD_LIBRARY_PATH", "")

        if self.config.lib_paths:
            merged = ":".join(map(str, self.config.lib_paths))
            if base_ld:
                merged = merged + ":" + base_ld
            self.child_env["LD_LIBRARY_PATH"] = merged
        else:
            self.child_env["LD_LIBRARY_PATH"] = base_ld

        self.runtime = Runtime(self.config, self.log, self.child_env)
        self.runtime.initialize_session()

        self.agents = Agents(
            coach_shm_id=self.runtime.coach_shm_id,
            trainer_shm_id=self.runtime.trainer_shm_id,
            player_shm_ids=self.runtime.player_shm_ids,
            config = self.config,
            log=self.log,
        )

        self.runtime.start_procs()

        self.last_state = None
        self.last_obs = None
        self.last_avail_actions = None
        self.done = 0
        self._need_restart = False
        self.episode_steps = 0
        self.turn_count = 0
        self.score = [0, 0]
        self._closed = False
        self.episode_limit = self.config.episode_limit

    def get_avail_actions(self):

        if self.done and self.last_avail_actions is not None:
            return self.last_avail_actions
        
        full_mask = self.agents.avail_actions()
        out = full_mask[:self.config.n1].copy()
        inactive_idx = np.flatnonzero(~self.agent_mask)

        if inactive_idx.size > 0:
            out[inactive_idx] = 0
            out[inactive_idx, self.agents.player_list[0].default_base_action] = 1

        self.last_avail_actions = out
        return self.last_avail_actions

    def reset(self):
        self.turn_count += 1
        self.log.info(f"Turn {self.turn_count}, Score={self.score}")
        if self._need_restart or (not self.runtime.has_live_procs()):
            self._need_restart = False
            self.agents.clear_all_shm_bufs()
            self.runtime.restart()

        self.last_state = None
        self.last_obs = None
        self.last_avail_actions = None
        self.done = 0
        self.episode_steps = 0

        goal = self.agents.coach.goal()

        if not self.agents.wait_all_ready():
            raise P.common.ShmProtocolError("Not READY Before Reset!!")

        if self.config.curriculum:
                self.agents.set_custom()
        else:
            if self.turn_count > 1 and int(goal) == 0:
                    self.agents.set_default()

        self.agents.coach.clear_goal_flag()

    def step(self, actions):

        self.episode_steps += 1

        if isinstance(actions, torch.Tensor):
            actions = actions.detach().cpu().numpy()

        actions = np.asarray(actions)
 
        self.agents.trainer.noop()
        self.agents.write_actions(actions, self.agent_mask)
        self._need_restart = not self.agents.wait_all_ready()
        timeout = (self.episode_steps >= self.episode_limit)
        goal = self.agents.coach.goal()
        reward = 0.0
        self.done = 0

        if timeout or self._need_restart:
            self.done = 1
        elif goal == 1:
            self.done = 1
            reward = 1.0
            self.score[0] += 1
        elif goal == -1:
            self.done = 1
            reward = -1.0
            self.score[1] += 1

        info = {
            "win": int(reward > 0),
            "lose": int(reward < 0),
            "timeout": int(timeout),
        }

        return float(reward), bool(self.done), info
    
    def get_obs(self):

        if self.done and self.last_obs is not None:
            return self.last_obs

        full_obs = self.agents.obs(norm=True)
        out = full_obs[:self.config.n1].copy()
        inactive_idx = np.flatnonzero(~self.agent_mask)
        for i in inactive_idx:
            out[i].fill(0.0)

        self.last_obs = out
        return self.last_obs

    def get_state(self):
        if self.done and self.last_state is not None:
            return self.last_state
        state = self.agents.state(norm=True)
        self.last_state = state.copy()
        return self.last_state

    def close(self):
        if self._closed:
            return
        self._closed = True
        self.runtime.close()
        self.agents.close()

    def get_env_info(self):
        return {
            "n_agents": int(self.config.n1),
            "n_actions": int(self.agents.n_actions),
            "state_shape": int(P.coach.COACH_STATE_FLOAT),
            "obs_shape": int(P.player.STATE_NUM),
            "episode_limit": int(self.episode_limit),
        }

