"""Phase 2 gate (part 2): the MVP learns a real task to non-trivial accuracy.

A small TRM trained on 4x4 Sudoku (the curriculum's first rung, BLUEPRINT section
10.2) must fill *blank* cells well above chance. Blank accuracy isolates learned
inference from trivially copying the given clues (chance on a 4x4 board is 0.25).
The trained model comes from the shared session fixture.
"""

import pytest
import torch

from eval.metrics import blank_accuracy, board_accuracy


@pytest.mark.slow
def test_mvp_learns_4x4_sudoku(trained_sudoku_4x4):
    d = trained_sudoku_4x4
    model, vx, vy = d["model"], d["vx"], d["vy"]

    with torch.no_grad():
        logits, _ = model(vx, height=d["height"], width=d["width"])
        pred = logits.argmax(-1)

    blank_acc = blank_accuracy(pred, vy, vx)
    board_acc = board_accuracy(pred, vy)
    print(f"\n4x4 Sudoku MVP: blank_acc={blank_acc:.3f} board_acc={board_acc:.3f}")

    # Non-trivial: well above the 0.25 random-guess baseline on blank cells.
    assert blank_acc > 0.5, f"blank_acc={blank_acc:.3f}, board_acc={board_acc:.3f}"
