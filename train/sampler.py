"""Checkpointable deterministic batch sampling for exact CPU resumption."""
from __future__ import annotations

from typing import Iterator

import torch
from torch.utils.data import Sampler


SAMPLER_STATE_VERSION = 1


class StatefulBatchSampler(Sampler[list[int]]):
    """Random-without-replacement batches with explicit resumable state.

    The sampler owns a private ``torch.Generator`` so batch order does not consume
    the global model/training RNG. With ``num_workers=0`` the saved ``position``
    points to the next batch exactly, which is the M04 deterministic CPU contract.
    """

    def __init__(
        self,
        dataset_size: int,
        batch_size: int,
        seed: int,
        *,
        drop_last: bool = False,
    ) -> None:
        if dataset_size <= 0:
            raise ValueError("dataset_size must be positive")
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        self.dataset_size = int(dataset_size)
        self.batch_size = int(batch_size)
        self.seed = int(seed)
        self.drop_last = bool(drop_last)
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(self.seed)
        self.epoch = -1
        self.position = 0
        self.order: torch.Tensor | None = None

    def _start_epoch(self) -> None:
        self.epoch += 1
        self.order = torch.randperm(self.dataset_size, generator=self.generator)
        self.position = 0

    def __iter__(self) -> Iterator[list[int]]:
        if self.order is None or self.position >= self.dataset_size:
            self._start_epoch()
        assert self.order is not None

        while self.position < self.dataset_size:
            end = min(self.position + self.batch_size, self.dataset_size)
            if self.drop_last and end - self.position < self.batch_size:
                self.position = self.dataset_size
                break
            batch = self.order[self.position:end].tolist()
            self.position = end
            yield batch

    def __len__(self) -> int:
        if self.drop_last:
            return self.dataset_size // self.batch_size
        return (self.dataset_size + self.batch_size - 1) // self.batch_size

    def state_dict(self) -> dict:
        return {
            "version": SAMPLER_STATE_VERSION,
            "dataset_size": self.dataset_size,
            "batch_size": self.batch_size,
            "seed": self.seed,
            "drop_last": self.drop_last,
            "epoch": self.epoch,
            "position": self.position,
            "generator_state": self.generator.get_state().clone(),
            "order": None if self.order is None else self.order.clone(),
        }

    def load_state_dict(self, state: dict) -> None:
        if not isinstance(state, dict) or state.get("version") != SAMPLER_STATE_VERSION:
            raise ValueError("unsupported sampler-state version")
        for key, expected in (
            ("dataset_size", self.dataset_size),
            ("batch_size", self.batch_size),
            ("seed", self.seed),
            ("drop_last", self.drop_last),
        ):
            if state.get(key) != expected:
                raise ValueError(
                    f"sampler {key} mismatch: checkpoint={state.get(key)!r}, current={expected!r}"
                )
        epoch = int(state["epoch"])
        position = int(state["position"])
        if epoch < -1 or not 0 <= position <= self.dataset_size:
            raise ValueError("invalid sampler epoch/position")
        order = state.get("order")
        if order is not None:
            order = torch.as_tensor(order, dtype=torch.int64, device="cpu").clone()
            if tuple(order.shape) != (self.dataset_size,):
                raise ValueError("sampler permutation has wrong shape")
            if not torch.equal(torch.sort(order).values, torch.arange(self.dataset_size)):
                raise ValueError("sampler permutation is not a complete index permutation")
        elif epoch >= 0:
            raise ValueError("started sampler state is missing its permutation")

        generator_state = torch.as_tensor(
            state["generator_state"], dtype=torch.uint8, device="cpu"
        ).clone()
        self.generator.set_state(generator_state)
        self.epoch = epoch
        self.position = position
        self.order = order
