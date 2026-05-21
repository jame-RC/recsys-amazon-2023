"""TrainingMonitor — logs metrics to JSON for the dashboard server to read."""

import json
import time
from pathlib import Path
from datetime import datetime


class TrainingMonitor:
    """Drop-in monitor that writes training metrics to a JSON file.

    Usage:
        monitor = TrainingMonitor("training_metrics.json")
        monitor.on_start(comment="My experiment")
        for epoch in range(num_epochs):
            loss = train_one_epoch()
            monitor.log(epoch=epoch, loss=loss, lr=current_lr)
        monitor.on_end(best_loss=best)
    """

    def __init__(self, log_file: str = "training_metrics.json"):
        self.log_file = Path(log_file)
        self._metrics: list[dict] = []
        self._start_time: float | None = None
        self._load_existing()

    def _load_existing(self) -> None:
        if self.log_file.exists():
            try:
                self._metrics = json.loads(self.log_file.read_text())
            except (json.JSONDecodeError, OSError):
                self._metrics = []

    def on_start(self, **info) -> None:
        """Call once when training begins."""
        self._start_time = time.time()
        entry = {"event": "start", "time": datetime.now().isoformat(), **info}
        self._metrics.append(entry)
        self._save()

    def log(self, epoch=None, **metrics) -> None:
        """Log one measurement point (one per epoch or per N batches).

        Pass epoch= as the epoch/call number, then any scalar metrics
        as keyword arguments (loss=..., accuracy=..., lr=... etc.)
        """
        entry = {
            "time": datetime.now().isoformat(),
            "elapsed": time.time() - self._start_time if self._start_time else 0,
            **( {"epoch": epoch} if epoch is not None else {} ),
            **metrics,
        }
        self._metrics.append(entry)
        self._save()

    def on_end(self, **info) -> None:
        """Call once when training finishes."""
        entry = {
            "event": "end",
            "time": datetime.now().isoformat(),
            "elapsed": time.time() - self._start_time if self._start_time else 0,
            **info,
        }
        self._metrics.append(entry)
        self._save()

    def _save(self) -> None:
        self.log_file.write_text(
            json.dumps(self._metrics, indent=2, ensure_ascii=False)
        )
