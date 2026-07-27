import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from code.train_nn import save_history_to_csv


def test_save_history_to_csv(tmp_path):
    output_path = tmp_path / "history.csv"
    history_rows = [
        {"epoch": 1, "train_loss": 0.75, "val_loss": 0.45, "lr": 0.0001},
    ]

    save_history_to_csv(history_rows, output_path)

    with output_path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))

    assert rows[0]["epoch"] == "1"
    assert rows[0]["train_loss"] == "0.75"
    assert rows[0]["val_loss"] == "0.45"
    assert rows[0]["lr"] == "0.0001"
