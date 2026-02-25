from __future__ import annotations

from typing import Dict, Tuple, List, Sequence
import numpy as np
from .protocols import P
from . import ipc
import time
from collections import Counter
from collections import Counter, defaultdict

Flags = Tuple[int, int]

class Agents:

    def __init__(
        self,
        *,
        config,
        coach_shm_id: str,
        trainer_shm_id: str,
        player_shm_ids: Dict[Tuple[int, int], str],
        log=None,
    ):
        self.log = log
        self.config = config
        self.coach_shm_id = coach_shm_id
        self.trainer_shm_id = trainer_shm_id
        self.player_shm_ids = dict(player_shm_ids)
        self.DEFAULT_BALL =  (0.0, 0.0, 0.0, 0.0)
        self.DEFAULT_LEFT_PLAYERS = [
            (-50.0,   0.0,   0.0, 0.0, 0.0),   # GK

            (-35.0, -18.0,   0.0, 0.0, 0.0),   # LB
            (-35.0,  -6.0,   0.0, 0.0, 0.0),   # LCB
            (-35.0,   6.0,   0.0, 0.0, 0.0),   # RCB
            (-35.0,  18.0,   0.0, 0.0, 0.0),   # RB

            (-20.0, -20.0,   0.0, 0.0, 0.0),   # LM
            (-20.0,  -7.0,   0.0, 0.0, 0.0),   # LCM
            (-20.0,   7.0,   0.0, 0.0, 0.0),   # RCM
            (-20.0,  20.0,   0.0, 0.0, 0.0),   # RM

            (-5.0,   -8.0,   0.0, 0.0, 0.0),   # ST1
            (-5.0,    8.0,   0.0, 0.0, 0.0),   # ST2
        ]        
        self.DEFAULT_RIGHT_PLAYERS = [
            ( 50.0,   0.0, 180.0, 0.0, 0.0),   # GK

            ( 35.0, -18.0, 180.0, 0.0, 0.0),
            ( 35.0,  -6.0, 180.0, 0.0, 0.0),
            ( 35.0,   6.0, 180.0, 0.0, 0.0),
            ( 35.0,  18.0, 180.0, 0.0, 0.0),

            ( 20.0, -20.0, 180.0, 0.0, 0.0),
            ( 20.0,  -7.0, 180.0, 0.0, 0.0),
            ( 20.0,   7.0, 180.0, 0.0, 0.0),
            ( 20.0,  20.0, 180.0, 0.0, 0.0),

            (  5.0,  -8.0, 180.0, 0.0, 0.0),
            (  5.0,   8.0, 180.0, 0.0, 0.0),
        ]
        self.DEFAULT_BODY_ANGLES = np.array(
            [0.0]*11 + [180.0]*11,
            dtype=np.float32
        )        
        self.CUSTOM_BALL =  (0.0, 0.0, 0.0, 0.0)
        self.CUSTOM_LEFT_PLAYERS: Sequence[Tuple[float, float, float, float, float]] =  [(-49.4387, 0.029, -90.207, 0.0, 0.0), (-1.8956, -21.2768, -12.76, 0.0, 0.0), (-1.0371, -4.6357, 172.306, 0.0, 0.0), (17.7481, -34.0658, 160.413, 0.0, 0.0), (7.1177, 8.723, 172.359, 0.0, 0.0), (10.6807, -13.5384, 171.579, 0.0, 0.0), (22.3294, -19.5177, -177.306, 0.0, 0.0), (23.066, -2.1292, 170.898, 0.0, 0.0), (36.7086, -31.193, -171.773, 0.0, 0.0), (32.6056, 1.8079, 145.407, 0.0, 0.0), (35.4272, -15.268, -179.539, 0.0, 0.0)]
        self.CUSTOM_RIGHT_PLAYERS: Sequence[Tuple[float, float, float, float, float]] =  [(49.7485, -4.62, -89.45, 0.0, 0.0), (0.5068, 0.7291, -154.191, 0.0, 0.0), (3.1451, -14.8675, -172.486, -0.6315, -0.09), (-10.1413, 9.4502, -135.735, 0.0, 0.0), (0.2775, -25.688, -176.48, 0.0, 0.0), (-19.6433, -12.3095, 149.132, -0.0427, 0.018), (-32.7386, -2.1781, 159.54, -0.913, 0.3405), (-19.277, -21.7438, 158.069, -0.8697, 0.3492), (-38.6289, 8.4134, -172.946, -0.967, -0.1197), (-40.6348, -25.915, 113.017, -0.385, 0.8965), (-39.8167, -14.8103, -158.944, -0.0398, -0.0137)]
        self.CUSTOM_BODY_ANGLES = np.array([-90.207, -12.76, 172.306, 160.413, 172.359, 171.579,-177.306, 170.898, -171.773, 145.407, -179.539,-89.45, -154.191, -172.486, -135.735, -176.48,149.132, 159.54, 158.069, -172.946, 113.017, -158.944],dtype=np.float32,)
        (
            self.coach_shms,
            self.trainer_shms,
            self.player_shms,) = ipc.create_shm_group(
            coach_names=[self.coach_shm_id],
            trainer_names=[self.trainer_shm_id],
            player_names=list(self.player_shm_ids.values()),
            coach_size=P.coach.COACH_SHM_SIZE,
            trainer_size=P.trainer.TRAINER_SHM_SIZE,
            player_size=P.player.PLAYER_SHM_SIZE,
            zero_fill=True,
            log=log,
        )

        self.coach = P.coach.Coach(self.coach_shms[self.coach_shm_id].buf)
        self.trainer = P.trainer.Trainer(self.trainer_shms[self.trainer_shm_id].buf)
        self.players: Dict[Tuple[int, int], P.player.Player] = {}

        for key, shm_id in self.player_shm_ids.items():
            buf = self.player_shms[shm_id].buf
            self.players[key] = P.player.Player(buf)

        # Ordered player list
        self.player_list: List[P.player.Player] = [
            self.players[(team, unum)]
            for team in (1, 2)
            for unum in range(
                1,
                self.config.n1 + 1 if team == 1 else self.config.n2 + 1,
            )
        ]

        if self.config.team1 == "hybrid":
            self.n_actions = P.player.HYBRID_MASK_NUM
        else:
            self.n_actions = P.player.BASE_MASK_NUM

        self._obs_buf = np.empty((len(self.player_list), P.player.STATE_NUM),dtype=np.float32,)
        self._mask_buf = np.empty((len(self.player_list), self.n_actions),dtype=np.int32,)
        
    def get_player(self, team: int, unum: int) -> P.player.Player:
        return self.players[(team, unum)]

    def all_players(self) -> List[P.player.Player]:
        return self.player_list

    def request(self, target: str) -> None:
        """
        Send REQUEST signal to:
        - "trainer"
        - "player:team:unum"
        """

        if target == "trainer":
            self.trainer.write_request()

        elif target.startswith("player:"):
            try:
                _, team, unum = target.split(":")
                team = int(team)
                unum = int(unum)
            except Exception:
                raise ValueError(
                    "player format must be 'player:team:unum'"
                )

            player = self.get_player(team, unum)
            player.write_request()

        else:
            raise ValueError(f"Unknown request target: {target}")

    def clear_all_shm_bufs(self) -> None:
        ipc.zero_all_shm_bufs(
            self.coach_shms,
            self.trainer_shms,
            self.player_shms,
        )

    def close(self) -> None:
        """
        Close and unlink shm objects.
        """

        self.clear_all_shm_bufs()

        for shm_dict in (
            self.coach_shms,
            self.trainer_shms,
            self.player_shms,
        ):
            for shm in shm_dict.values():
                try:
                    shm.close()
                except Exception:
                    pass
                try:
                    shm.unlink()
                except Exception:
                    pass

    def state(self, norm: bool = True):
        """
        Return current global state from coach.

        norm=True  -> normalized state
        norm=False -> raw state
        """
        if norm:
            return self.coach.state_norm(half_field_length=self.config.half_length, half_field_width=self.config.half_width)
        else:
            return self.coach.state()

    def obs(self, norm: bool = True):
        """
        Return stacked observations of all players.
        Shape: (n_total_players, STATE_NUM)
        """

        for i, p in enumerate(self.player_list):
            if norm:
                self._obs_buf[i] = p.obs_norm(
                    half_field_length=self.config.half_length,
                    half_field_width=self.config.half_width,
                )
            else:
                self._obs_buf[i] = p.obs()

        return self._obs_buf
    
    def avail_actions(self):
        for i, p in enumerate(self.player_list):
            if self.config.team1 == "hybrid":
                self._mask_buf[i] = p.hybrid_mask()
            else:
                self._mask_buf[i] = p.base_mask()

        return self._mask_buf

    def write_base_actions(self, actions: np.ndarray, agent_mask: np.ndarray):
        """
        actions: (n1,) team1 动作
        agent_mask: (n1,) bool
        """

        n1 = self.config.n1

        for idx, p in enumerate(self.player_list):

            if idx < n1:  # team1
                if agent_mask[idx]:
                    act = int(actions[idx])
                else:
                    act = p.default_base_action
            else:         # team2
                act = p.default_base_action

            p.write_base_action(act)
            p.write_request()

    def write_hybrid_actions(self, actions: np.ndarray, agent_mask: np.ndarray):
        """
        actions: (n1, 3)  -> [a, u0, u1]
        agent_mask: (n1,) bool
        """

        n1 = self.config.n1

        for idx, p in enumerate(self.player_list):

            if idx < n1:  # team1
                if agent_mask[idx]:
                    a  = int(actions[idx, 0])
                    u0 = float(actions[idx, 1])
                    u1 = float(actions[idx, 2])
                else:
                    a, u0, u1 = p.default_hybrid_action
            else:
                a, u0, u1 = p.default_hybrid_action

            p.write_hybrid_action(a, u0, u1)
            p.write_request()

    def write_actions(self, actions: np.ndarray, agent_mask: np.ndarray):
        if self.config.team1 == "hybrid":
            self.write_hybrid_actions(actions, agent_mask)
        else:
            self.write_base_actions(actions, agent_mask)

    def wait_all_ready(
        self,
        timeout: float = 36000.0,
        poll: float = 0.0005,
        stuck_window: float = 10.0,
        rescue_cooldown: float = 0.5,
    ):

        t_end = time.monotonic() + float(timeout)

        last_dist = None
        last_change_t = time.monotonic()
        last_rescue_t = 0.0

        total = len(self.player_list) + 1  # players + trainer

        while True:

            now = time.monotonic()

            pairs = []
            ready_entities = []

            # ---- check players ----
            for p in self.player_list:
                a, b = p.read_flags()
                ab = (int(a), int(b))
                pairs.append(ab)

                if ab == P.common.FLAG_READY:
                    ready_entities.append(("player", p))

            # ---- check trainer ----
            ta, tb = self.trainer.flags()
            tab = (int(ta), int(tb))
            pairs.append(tab)

            if tab == P.common.FLAG_READY:
                ready_entities.append(("trainer", self.trainer))

            # ---- all READY ----
            if len(ready_entities) == total:
                return True

            # ---- timeout ----
            if now >= t_end:
                if self.log:
                    self.log.info(
                        f"[wait_all_ready] timeout dist={dict(Counter(pairs))}"
                    )
                return False

            # ---- steady detection ----
            dist = Counter(pairs)
            if dist != last_dist:
                last_dist = dist
                last_change_t = now

            stuck = (now - last_change_t) >= float(stuck_window)

            # ---- rescue ----
            if stuck and ready_entities and (now - last_rescue_t) >= rescue_cooldown:

                pushed = 0

                for kind, obj in ready_entities:

                    if kind == "trainer":
                        obj.noop()
                    else:
                        obj.take_default_action(is_hybrid=(self.config.team1 == "hybrid"))

                    pushed += 1

                last_rescue_t = now
                last_change_t = now

            time.sleep(float(poll))

    def write_all_body_targets(self, angles: np.ndarray):
        if len(angles) != len(self.player_list):
            raise ValueError(
                f"angles size mismatch: got {len(angles)}, expected {len(self.player_list)}"
            )
        for idx, p in enumerate(self.player_list):
            p.write_body_target_deg(float(angles[idx]))
            
    def set_default(self) -> None:

        self.trainer.reset_players_and_ball(
            self.DEFAULT_BALL,
            self.DEFAULT_LEFT_PLAYERS,
            self.DEFAULT_RIGHT_PLAYERS,
        )
        self.write_all_body_targets(self.DEFAULT_BODY_ANGLES)
        is_hybrid = (self.config.team1 == "hybrid")
        for p in self.player_list:
            p.take_empty_action(is_hybrid=is_hybrid)
        self.wait_all_ready()

    def set_custom(self) -> None:

        self.trainer.reset_players_and_ball(
            self.CUSTOM_BALL,
            self.CUSTOM_LEFT_PLAYERS,
            self.CUSTOM_RIGHT_PLAYERS,
        )
        self.write_all_body_targets(self.CUSTOM_BODY_ANGLES)
        is_hybrid = (self.config.team1 == "hybrid")
        for p in self.player_list:
            p.take_empty_action(is_hybrid=is_hybrid)
        self.wait_all_ready()

    def read_all_flags(self, include_cycles: bool = False) -> Dict[str, object]:
        who: Dict[str, object] = {}
        groups: Dict[Flags, List[str]] = defaultdict(list)
        pairs: List[Flags] = []

        # ---- players ----
        for i, p in enumerate(self.player_list):
            ab: Flags = tuple(map(int, p.read_flags()))
            name = f"player:{i}"
            pairs.append(ab)
            groups[ab].append(name)

            if include_cycles:
                try:
                    cyc = int(p.cycle())
                except Exception:
                    cyc = None
                who[name] = {"flags": ab, "cycle": cyc}
            else:
                who[name] = ab

        # ---- trainer ----
        tab: Flags = tuple(map(int, self.trainer.flags()))
        pairs.append(tab)
        groups[tab].append("trainer")
        who["trainer"] = {"flags": tab, "cycle": None} if include_cycles else tab

        return {
            "who": who,
            "groups": dict(groups),
            "dist": Counter(pairs),
            "trainer": tab,
        }