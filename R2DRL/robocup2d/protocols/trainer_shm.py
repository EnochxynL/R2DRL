from __future__ import annotations

from typing import Final, Sequence, Tuple, Union
import struct
import time

from .common import align4, FLAG_READY, FLAG_REQ, _I32, _F32

Buf = Union[memoryview, bytearray]

# ===================== Trainer SHM layout =====================

TRAINER_SHM_SIZE: Final[int] = 4096

T_FLAG_A: Final[int] = 0
T_FLAG_B: Final[int] = 1
T_OPCODE: Final[int] = 4

T_BALL_X:  Final[int] = align4(T_OPCODE + 4)
T_BALL_Y:  Final[int] = T_BALL_X + 4
T_BALL_VX: Final[int] = T_BALL_Y + 4
T_BALL_VY: Final[int] = T_BALL_VX + 4

N_LEFT: Final[int] = 11
N_RIGHT: Final[int] = 11
PLAYER_STRIDE: Final[int] = 5 * 4

T_PLAYERS_BASE: Final[int] = align4(T_BALL_VY + 4)

def T_LPX(i: int) -> int: return T_PLAYERS_BASE + i * PLAYER_STRIDE + 0 * 4
def T_LPY(i: int) -> int: return T_PLAYERS_BASE + i * PLAYER_STRIDE + 1 * 4
def T_LPD(i: int) -> int: return T_PLAYERS_BASE + i * PLAYER_STRIDE + 2 * 4
def T_LVX(i: int) -> int: return T_PLAYERS_BASE + i * PLAYER_STRIDE + 3 * 4
def T_LVY(i: int) -> int: return T_PLAYERS_BASE + i * PLAYER_STRIDE + 4 * 4

T_R_BASE: Final[int] = T_PLAYERS_BASE + N_LEFT * PLAYER_STRIDE

def T_RPX(i: int) -> int: return T_R_BASE + i * PLAYER_STRIDE + 0 * 4
def T_RPY(i: int) -> int: return T_R_BASE + i * PLAYER_STRIDE + 1 * 4
def T_RPD(i: int) -> int: return T_R_BASE + i * PLAYER_STRIDE + 2 * 4
def T_RVX(i: int) -> int: return T_R_BASE + i * PLAYER_STRIDE + 3 * 4
def T_RVY(i: int) -> int: return T_R_BASE + i * PLAYER_STRIDE + 4 * 4

OP_NOOP: Final[int] = 0
OP_RESET_RANDOMLY: Final[int] = 1
OP_RESET_FROM_PY: Final[int] = 2
OP_PLAY_ON: Final[int] = 3


TRAINER_WAIT_READY_TIMEOUT_MS: Final[int] = 30000
TRAINER_WAIT_DONE_TIMEOUT_MS:  Final[int] = 30000
TRAINER_POLL_US:               Final[int] = 100

def assert_trainer_shm_size(size: int) -> None:
    if int(size) != int(TRAINER_SHM_SIZE):
        raise RuntimeError(
            f"trainer shm size mismatch: got={size} expected={TRAINER_SHM_SIZE}"
        )


# ===================== Trainer View =====================

class Trainer:

    def __init__(self, buf: Buf):
        self.buf = buf

    # ================= flags =================

    def flags(self) -> Tuple[int, int]:
        return int(self.buf[T_FLAG_A]), int(self.buf[T_FLAG_B])

    def write_request(self) -> None:
        """
        Set flags to REQUEST (1,0).
        Order must match C++: write B first, then A.
        """
        a, b = FLAG_REQ
        self.buf[T_FLAG_B] = int(b) & 0xFF
        self.buf[T_FLAG_A] = int(a) & 0xFF

    def wait_ready(
        self,
        timeout_ms: int = TRAINER_WAIT_READY_TIMEOUT_MS,
        poll_us: int = TRAINER_POLL_US,
    ) -> bool:

        deadline = time.monotonic() + timeout_ms / 1000.0

        while time.monotonic() < deadline:
            if self.flags() == FLAG_READY:
                return True
            time.sleep(poll_us / 1_000_000.0)

        return False

    # ================= opcode =================

    def write_opcode(self, opcode: int) -> None:
        struct.pack_into(_I32, self.buf, T_OPCODE, int(opcode))

    def submit_opcode(self, opcode: int) -> None:
        self.write_opcode(opcode)
        self.write_request()

    # ================= payload =================

    def write_ball(self, bx, by, bvx, bvy):
        struct.pack_into(_F32, self.buf, T_BALL_X, float(bx))
        struct.pack_into(_F32, self.buf, T_BALL_Y, float(by))
        struct.pack_into(_F32, self.buf, T_BALL_VX, float(bvx))
        struct.pack_into(_F32, self.buf, T_BALL_VY, float(bvy))

    def write_left_players(self, left_players: Sequence[Tuple[float, float, float, float, float]]):
        if len(left_players) != N_LEFT:
            raise ValueError
        for i, (x, y, deg, vx, vy) in enumerate(left_players):
            struct.pack_into(_F32, self.buf, T_LPX(i), float(x))
            struct.pack_into(_F32, self.buf, T_LPY(i), float(y))
            struct.pack_into(_F32, self.buf, T_LPD(i), float(deg))
            struct.pack_into(_F32, self.buf, T_LVX(i), float(vx))
            struct.pack_into(_F32, self.buf, T_LVY(i), float(vy))

    def write_right_players(self, right_players: Sequence[Tuple[float, float, float, float, float]]):
        if len(right_players) != N_RIGHT:
            raise ValueError
        for i, (x, y, deg, vx, vy) in enumerate(right_players):
            struct.pack_into(_F32, self.buf, T_RPX(i), float(x))
            struct.pack_into(_F32, self.buf, T_RPY(i), float(y))
            struct.pack_into(_F32, self.buf, T_RPD(i), float(deg))
            struct.pack_into(_F32, self.buf, T_RVX(i), float(vx))
            struct.pack_into(_F32, self.buf, T_RVY(i), float(vy))

    def write_reset_payload(
        self,
        ball: Tuple[float, ...],
        left_players: Sequence[Tuple[float, float, float, float, float]],
        right_players: Sequence[Tuple[float, float, float, float, float]],
    ):
        if len(ball) == 2:
            bx, by = ball
            bvx, bvy = 0.0, 0.0
        elif len(ball) == 4:
            bx, by, bvx, bvy = ball
        else:
            raise ValueError

        self.write_ball(bx, by, bvx, bvy)
        self.write_left_players(left_players)
        self.write_right_players(right_players)

    def reset_players_and_ball(self, ball, left_players, right_players) -> None:
        """
        Write reset payload (ball + players) and trigger OP_RESET_FROM_PY.
        """
        self.write_reset_payload(ball, left_players, right_players)
        self.submit_opcode(OP_RESET_FROM_PY)

    def noop(self) -> None:
        self.submit_opcode(OP_NOOP)