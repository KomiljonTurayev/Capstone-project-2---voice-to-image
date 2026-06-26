# logger.py
import logging
from datetime import datetime

class PipelineLogger:
    def __init__(self):
        logging.basicConfig(format="%(message)s", level=logging.INFO)
        self._logger = logging.getLogger("pipeline")

    def step(self, n: int, total: int, name: str, model: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._logger.info(f"[{ts}] ── STEP {n}/{total} ─ {name:<35} (model: {model})")

    def done(self, message: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._logger.info(f"[{ts}] ✓  {message}")

    def error(self, step: str, error: str) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._logger.error(f"[{ts}] ✗  {step} failed: {error}")

    def finish(self, duration: float) -> None:
        ts = datetime.now().strftime("%H:%M:%S")
        self._logger.info(f"[{ts}] ── DONE ── Total: {duration:.1f}s")

log = PipelineLogger()
