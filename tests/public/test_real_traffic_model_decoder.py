from __future__ import annotations

import itertools
import os
import random

import pytest

from experiments.real_traffic.fair_benchmark import model_to_labels
from experiments.real_traffic.model_decoder import (
    ModelDecoderRuntime,
    build_model_decoder,
)


@pytest.fixture(scope="session")
def decoder_runtime(tmp_path_factory):
    override = os.environ.get("SPECTRA_MODEL_DECODER_LIBRARY")
    path = (override if override is not None else
            build_model_decoder(tmp_path_factory.mktemp("model-decoder")))
    return ModelDecoderRuntime(path)


def signed_model(bits):
    return [index + 1 if bit else -(index + 1)
            for index, bit in enumerate(bits)]


def test_exhaustive_decoder_matches_python_reference(decoder_runtime) -> None:
    masks = (0b0001, 0b0011, 0b0110, 0b1100, 1 << 63)
    with decoder_runtime.prepare(masks) as decoder:
        assert decoder.info["variables"] == len(masks)
        assert decoder.info["payload_bytes"] == 2 * len(masks)
        for bits in itertools.product((False, True), repeat=len(masks)):
            model = signed_model(bits)
            expected = bytes(model_to_labels(model, masks))
            assert decoder.decode(model) == expected
            assert decoder.decode(tuple(model)) == expected


def test_random_partial_models_keep_unassigned_variables_low(decoder_runtime) -> None:
    rng = random.Random(316477)
    for _ in range(500):
        variables = rng.randrange(1, 50)
        masks = []
        for _variable in range(variables):
            first = rng.randrange(64)
            if rng.randrange(3):
                second = rng.randrange(64)
                masks.append((1 << first) | (1 << second))
            else:
                masks.append(1 << first)
        masks = tuple(masks)
        model = []
        for variable in range(variables):
            if rng.randrange(3):
                model.append(variable + 1 if rng.randrange(2) else -(variable + 1))
        with decoder_runtime.prepare(masks) as decoder:
            observed = decoder.decode(model)
        positives = {literal for literal in model if literal > 0}
        expected = []
        for variable, mask in enumerate(masks):
            choices = [color for color in range(64) if mask >> color & 1]
            expected.append(choices[-1] if variable + 1 in positives else choices[0])
        assert observed == bytes(expected)


def test_empty_partial_model_uses_every_low_choice(decoder_runtime) -> None:
    with decoder_runtime.prepare((0b0011, 0b1100, 1 << 63)) as decoder:
        assert decoder.decode([]) == bytes((0, 2, 63))


@pytest.mark.parametrize("bad", [[0], [1.0], [1, 1], [1, -1], [3]])
def test_invalid_model_is_rejected(decoder_runtime, bad) -> None:
    with decoder_runtime.prepare((0b11,)) as decoder:
        with pytest.raises(ValueError):
            decoder.decode(bad)


@pytest.mark.parametrize("masks", [[], [1], (0,), (0b111,), (True,), (1 << 64,)])
def test_invalid_masks_are_rejected(decoder_runtime, masks) -> None:
    with pytest.raises(ValueError):
        decoder_runtime.prepare(masks)  # type: ignore[arg-type]


def test_payload_cap_and_closed_lifecycle(decoder_runtime) -> None:
    with pytest.raises(MemoryError):
        decoder_runtime.prepare((0b11,) * 100, max_bytes=10)
    prepared = decoder_runtime.prepare((0b11, 0b1100))
    assert prepared.decode([1, -2]) == bytes((1, 2))
    prepared.close()
    prepared.close()
    with pytest.raises(RuntimeError):
        prepared.decode([1, -2])
    with pytest.raises(RuntimeError):
        _ = prepared.info
    with pytest.raises(RuntimeError):
        prepared.__enter__()


def test_explicit_builder_does_not_overwrite(tmp_path) -> None:
    (tmp_path / "build.json").write_text("preserve")
    with pytest.raises(FileExistsError):
        build_model_decoder(tmp_path)
    assert (tmp_path / "build.json").read_text() == "preserve"


def test_missing_compiler_has_no_fallback(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        build_model_decoder(tmp_path, compiler="missing-model-decoder-compiler")
