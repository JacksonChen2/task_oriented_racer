"""Per-source cardinality constraint for BS-assisted UAV relay actions."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RelayFanoutStatus:
    counts: np.ndarray
    overflow: np.ndarray
    cost: float

    @property
    def max_count(self) -> int:
        return int(np.max(self.counts, initial=0))

    @property
    def violating_senders(self) -> int:
        return int(np.count_nonzero(self.overflow))


@dataclass(frozen=True)
class RelayFanoutConstraint:
    """Limit how many receivers one information owner can reach via the BS.

    Rows ``0..N-1`` of the scheduler matrix identify the information owner,
    and columns identify receivers.  The upload row is deliberately excluded
    from this constraint.

    The CRPO cost is the worst normalized overflow in the *proposed* action.
    A separate projection provides the hard physical guarantee.  This gives
    CRPO a useful violation signal without ever forwarding to receiver K+1.
    """

    n_uavs: int
    max_recipients: int = 4

    def __post_init__(self) -> None:
        if self.n_uavs < 1:
            raise ValueError("n_uavs must be positive")
        if not 0 <= self.max_recipients <= max(0, self.n_uavs - 1):
            raise ValueError(
                "max_recipients must be between 0 and n_uavs - 1"
            )

    def _validated(self, matrix: np.ndarray) -> np.ndarray:
        value = np.asarray(matrix, dtype=np.int8)
        expected = (self.n_uavs + 1, self.n_uavs)
        if value.shape != expected:
            raise ValueError(
                f"relay action matrix must be {expected}, got {value.shape}"
            )
        if np.any((value != 0) & (value != 1)):
            raise ValueError("relay action matrix must contain only 0 or 1")
        if np.any(np.diag(value[: self.n_uavs])):
            raise ValueError("relay action diagonal must be zero")
        return value

    def evaluate(self, matrix: np.ndarray) -> RelayFanoutStatus:
        value = self._validated(matrix)
        counts = np.sum(value[: self.n_uavs], axis=1, dtype=np.int64)
        overflow = np.maximum(counts - self.max_recipients, 0)
        possible_overflow = self.n_uavs - 1 - self.max_recipients
        cost = (
            0.0
            if possible_overflow <= 0
            else float(np.max(overflow, initial=0) / possible_overflow)
        )
        return RelayFanoutStatus(counts.copy(), overflow.copy(), cost)

    def project(
        self, matrix: np.ndarray, priority: np.ndarray | None = None
    ) -> np.ndarray:
        """Return a valid matrix, preserving uploads and highest priorities."""

        value = self._validated(matrix)
        output = value.copy()
        if priority is None:
            scores = np.zeros((self.n_uavs, self.n_uavs), dtype=np.float64)
        else:
            scores = np.asarray(priority, dtype=np.float64)
            if scores.shape != (self.n_uavs, self.n_uavs):
                raise ValueError(
                    "relay priority must have shape "
                    f"({self.n_uavs}, {self.n_uavs})"
                )
            if not np.all(np.isfinite(scores)):
                raise ValueError("relay priority contains NaN or Inf")

        for sender in range(self.n_uavs):
            selected = np.flatnonzero(output[sender])
            if selected.size <= self.max_recipients:
                continue
            # Stable receiver-id tie breaking makes replay deterministic.
            keep = sorted(
                selected.tolist(),
                key=lambda receiver: (-scores[sender, receiver], receiver),
            )[: self.max_recipients]
            output[sender, :] = 0
            output[sender, keep] = 1
        return output

    def validate_executed(self, matrix: np.ndarray) -> None:
        status = self.evaluate(matrix)
        if status.violating_senders:
            raise ValueError(
                "executed BS relay action exceeds per-UAV receiver limit: "
                f"counts={status.counts.tolist()}, limit={self.max_recipients}"
            )
