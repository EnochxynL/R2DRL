from __future__ import annotations

import struct
from dataclasses import dataclass
from typing import Final, Union, Tuple

import numpy as np
from .common import align4, clamp01, FLAG_READY, FLAG_REQ, _U8, _I32, _F32
import time

Buf = Union[memoryview, bytearray]

# =============================================================================
# Player SHM layout (must match C++ SamplePlayer exactly)
# Fixed 11-player template:
#   self : 6
#   ball : 4
#   opp  : 11 * 4
#   mate : 10 * 4   (excluding self)
#   tail : 3
# =============================================================================

STATE_SELF_DIM: Final[int] = 6
STATE_BALL_DIM: Final[int] = 4
STATE_PLAYER_DIM: Final[int] = 4
STATE_OPP_SLOTS: Final[int] = 11
STATE_MATE_SLOTS: Final[int] = 10
STATE_TAIL_DIM: Final[int] = 3

STATE_NUM: Final[int] = (
    STATE_SELF_DIM
    + STATE_BALL_DIM
    + STATE_OPP_SLOTS * STATE_PLAYER_DIM
    + STATE_MATE_SLOTS * STATE_PLAYER_DIM
    + STATE_TAIL_DIM
)

BASE_MASK_NUM: Final[int] = 19
HYBRID_MASK_NUM: Final[int] = 6

OFFSET_FLAG_A: Final[int] = 0
OFFSET_FLAG_B: Final[int] = 1
OFFSET_MASK:   Final[int] = align4(OFFSET_FLAG_B + 1)

OFFSET_CYCLE:  Final[int] = align4(OFFSET_MASK + BASE_MASK_NUM)
OFFSET_STATE:  Final[int] = align4(OFFSET_CYCLE + 4)
OFFSET_ACTION: Final[int] = align4(OFFSET_STATE + STATE_NUM * 4)

OFFSET_HYBRID_MASK: Final[int] = align4(OFFSET_ACTION + 4)
OFFSET_HYBRID_ACT:  Final[int] = align4(OFFSET_HYBRID_MASK + HYBRID_MASK_NUM)
OFFSET_HYBRID_U0:   Final[int] = align4(OFFSET_HYBRID_ACT + 4)
OFFSET_HYBRID_U1:   Final[int] = align4(OFFSET_HYBRID_U0 + 4)

OFFSET_BODY_TARGET_DEG: Final[int] = align4(OFFSET_HYBRID_U1 + 4)
PLAYER_SHM_SIZE: Final[int] = align4(OFFSET_BODY_TARGET_DEG + 4)

# ---- struct formats ----
_MASK_BASE: Final[str] = f"<{BASE_MASK_NUM}B"
_MASK_HYBRID: Final[str] = f"<{HYBRID_MASK_NUM}B"

PLAYER_WAIT_READY_TIMEOUT_MS: Final[int] = 30000
PLAYER_WAIT_DONE_TIMEOUT_MS:  Final[int] = 30000
PLAYER_POLL_US:               Final[int] = 100

# =============================================================================
# Data containers
# =============================================================================

@dataclass(frozen=True)
class Flags:
    a: int
    b: int


@dataclass(frozen=True)
class HybridAction:
    a: int
    u0: float
    u1: float


# =============================================================================
# Basic helper
# =============================================================================

def assert_player_shm_size(size: int) -> None:
    if int(size) != int(PLAYER_SHM_SIZE):
        raise RuntimeError(
            f"player shm size mismatch: got={size} expected={PLAYER_SHM_SIZE}"
        )


# =============================================================================
# Player SHM View (Object-Oriented Access)
# =============================================================================

class Player:

    def __init__(self, buf: Buf):
        self.buf = buf
        self.default_base_action: int = 17
        self.default_hybrid_action: Tuple[int, float, float] = (4, 0.0, 0.0)
        self.empty_base_action: int = 18
        self.empty_hybrid_action: Tuple[int, float, float] = (5, 0.0, 0.0)

    # =============================
    # meta
    # =============================
    def cycle(self) -> int:
        return struct.unpack_from(_I32, self.buf, OFFSET_CYCLE)[0]

    def read_flags(self) -> Tuple[int, int]:
        return int(self.buf[OFFSET_FLAG_A]), int(self.buf[OFFSET_FLAG_B])

    # =============================
    # mask
    # =============================
    def base_mask(self) -> np.ndarray:
        raw = struct.unpack_from(_MASK_BASE, self.buf, OFFSET_MASK)
        return np.asarray(raw, dtype=np.int32)

    def hybrid_mask(self) -> np.ndarray:
        raw = struct.unpack_from(_MASK_HYBRID, self.buf, OFFSET_HYBRID_MASK)
        return np.asarray(raw, dtype=np.int32)

    # =============================
    # observation
    # =============================
    def obs(self, copy=True) -> np.ndarray:
        arr = np.frombuffer(
            self.buf,
            dtype=np.float32,
            count=STATE_NUM,
            offset=OFFSET_STATE,
        )
        return arr.copy() if copy else arr

    def obs_norm(
        self,
        half_field_length: float,
        half_field_width: float,
        stamina_max: float = 8000.0,
    ) -> np.ndarray:

        PLAYER_VMAX = 1.05
        BALL_VMAX = 3.0

        o = self.obs(copy=True)

        HL = float(half_field_length)
        HW = float(half_field_width)

        SELF_BEG = 0
        SELF_END = SELF_BEG + STATE_SELF_DIM

        BALL_BEG = SELF_END
        BALL_END = BALL_BEG + STATE_BALL_DIM

        OPP_BEG = BALL_END
        OPP_END = OPP_BEG + STATE_OPP_SLOTS * STATE_PLAYER_DIM

        MATE_BEG = OPP_END
        MATE_END = MATE_BEG + STATE_MATE_SLOTS * STATE_PLAYER_DIM

        TAIL_BEG = MATE_END
        TAIL_END = TAIL_BEG + STATE_TAIL_DIM

        assert TAIL_END == STATE_NUM, (
            f"obs layout mismatch: tail_end={TAIL_END}, STATE_NUM={STATE_NUM}"
        )

        # --- self ---
        o[SELF_BEG + 0] /= HL
        o[SELF_BEG + 1] /= HW
        o[SELF_BEG + 2] /= PLAYER_VMAX
        o[SELF_BEG + 3] /= PLAYER_VMAX
        o[SELF_BEG + 4] = 1.0   # intentionally ignore stamina magnitude
        # o[SELF_BEG + 5] = kickable flag, keep as 0/1

        # --- ball ---
        o[BALL_BEG + 0] /= HL
        o[BALL_BEG + 1] /= HW
        o[BALL_BEG + 2] /= BALL_VMAX
        o[BALL_BEG + 3] /= BALL_VMAX

        # --- opponents: fixed 11 slots ---
        opp = o[OPP_BEG:OPP_END].reshape(STATE_OPP_SLOTS, STATE_PLAYER_DIM)
        opp[:, 0] /= HL
        opp[:, 1] /= HW
        opp[:, 2] /= PLAYER_VMAX
        opp[:, 3] /= PLAYER_VMAX

        # --- mates: fixed 10 slots ---
        mate = o[MATE_BEG:MATE_END].reshape(STATE_MATE_SLOTS, STATE_PLAYER_DIM)
        mate[:, 0] /= HL
        mate[:, 1] /= HW
        mate[:, 2] /= PLAYER_VMAX
        mate[:, 3] /= PLAYER_VMAX

        # tail (game mode / side / goalie) left unchanged
        return o

    # =============================
    # write action
    # =============================
    def write_base_action(self, act: int) -> None:
        struct.pack_into(_I32, self.buf, OFFSET_ACTION, int(act))

    def write_hybrid_action(
        self,
        a: int,
        u0: float,
        u1: float,
        clamp: bool = True,
    ) -> None:

        if clamp:
            u0 = clamp01(u0)
            u1 = clamp01(u1)

        struct.pack_into(_I32, self.buf, OFFSET_HYBRID_ACT, int(a))
        struct.pack_into(_F32, self.buf, OFFSET_HYBRID_U0, float(u0))
        struct.pack_into(_F32, self.buf, OFFSET_HYBRID_U1, float(u1))

    def wait_ready(
        self,
        timeout_ms: int = PLAYER_WAIT_READY_TIMEOUT_MS,
        poll_us: int = PLAYER_POLL_US,
    ) -> bool:

        deadline = time.monotonic() + timeout_ms / 1000.0

        while time.monotonic() < deadline:
            if self.read_flags() == FLAG_READY:
                return True
            time.sleep(poll_us / 1_000_000.0)

        return False

    def write_request(self) -> None:
        # 与 Trainer 保持一致：先写 B 再写 A
        self.buf[OFFSET_FLAG_B] = FLAG_REQ[1] & 0xFF
        self.buf[OFFSET_FLAG_A] = FLAG_REQ[0] & 0xFF

    def write_body_target_deg(self, angle_deg: float) -> None:
        """
        Write absolute body target angle (degree) into shared memory.
        C++ case 18 will read this value and compute turn moment.
        """
        struct.pack_into(_F32, self.buf, OFFSET_BODY_TARGET_DEG, float(angle_deg))

    def take_default_action(
        self,
        is_hybrid: bool,
    ) -> None:
        """
        Execute default fallback action and send request.
        """

        if is_hybrid:
            a, u0, u1 = self.default_hybrid_action
            self.write_hybrid_action(int(a), float(u0), float(u1))
        else:
            self.write_base_action(int(self.default_base_action))

        self.write_request()

    def take_empty_action(
        self,
        is_hybrid: bool,
    ) -> None:
        """
        Execute default fallback action and send request.
        """

        if is_hybrid:
            a, u0, u1 = self.empty_hybrid_action
            self.write_hybrid_action(int(a), float(u0), float(u1))
        else:
            self.write_base_action(int(self.empty_base_action))

        self.write_request()