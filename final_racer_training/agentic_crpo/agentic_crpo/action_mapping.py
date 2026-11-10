"""Canonical mapping between the N^2 policy bits and scheduler matrix."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ActionMapping:
    """Encode/decode the action ordering required by the scheduler.

    The first ``N(N-1)`` bits are ``B[i, j]`` in row-major order while
    skipping ``i == j``.  The final ``N`` bits are uploads ``u[i]``.  The
    matrix representation is ``(N + 1, N)`` with uploads in its last row.
    """

    n_uavs: int

    def __post_init__(self) -> None:
        if self.n_uavs < 1:
            raise ValueError("n_uavs must be positive")

    @property
    def action_dim(self) -> int:
        return self.n_uavs * self.n_uavs

    @property
    def relay_dim(self) -> int:
        return self.n_uavs * (self.n_uavs - 1)

    def decode(self, action: np.ndarray) -> np.ndarray:
        bits = np.asarray(action, dtype=np.int8).reshape(-1)
        if bits.shape != (self.action_dim,):
            raise ValueError(
                f"action shape must be ({self.action_dim},), got {bits.shape}"
            )
        if np.any((bits != 0) & (bits != 1)):
            raise ValueError("MultiBinary action must contain only 0 or 1")
        matrix = np.zeros((self.n_uavs + 1, self.n_uavs), dtype=np.int8)
        offset = 0
        for sender in range(self.n_uavs):
            for receiver in range(self.n_uavs):
                if sender == receiver:
                    continue
                matrix[sender, receiver] = bits[offset]
                offset += 1
        matrix[self.n_uavs, :] = bits[offset:]
        assert offset + self.n_uavs == self.action_dim
        return matrix

    def encode(self, matrix: np.ndarray) -> np.ndarray:
        value = np.asarray(matrix)
        expected = (self.n_uavs + 1, self.n_uavs)
        if value.shape != expected:
            raise ValueError(f"matrix shape must be {expected}, got {value.shape}")
        if np.any((value != 0) & (value != 1)):
            raise ValueError("action matrix must contain only 0 or 1")
        if np.any(np.diag(value[: self.n_uavs])):
            raise ValueError("relay diagonal B[i, i] must be zero")
        bits: list[int] = []
        for sender in range(self.n_uavs):
            for receiver in range(self.n_uavs):
                if sender != receiver:
                    bits.append(int(value[sender, receiver]))
        bits.extend(int(item) for item in value[self.n_uavs, :])
        return np.asarray(bits, dtype=np.int8)

    def split(self, action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        matrix = self.decode(action)
        return matrix[: self.n_uavs].copy(), matrix[self.n_uavs].copy()

