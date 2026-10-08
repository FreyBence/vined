"""Periodic console progress for long-running pipeline operations."""

import logging
from time import monotonic


logger = logging.getLogger("vined.progress")


def configure_progress():
    """Enable timestamped stderr progress for command-line entry points."""
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class Progress:
    """Report completed work at most every ten seconds, plus first/last items."""

    def __init__(self, label, total=None, unit="items"):
        self.label, self.total, self.unit = label, total, unit
        self.started = self.last_report = monotonic()
        self.count = 0
        logger.info("%s: starting%s", label, "" if total is None else f" ({total} {unit})")

    def update(self, count, detail=""):
        first = self.count == 0 and count > 0
        self.count = count
        now = monotonic()
        if first or count == self.total or now - self.last_report >= 10:
            elapsed = now - self.started
            completed = str(count) if self.total is None else f"{count}/{self.total}"
            remaining = (f", ETA {elapsed * (self.total - count) / count:.1f}s"
                         if self.total is not None and 0 < count < self.total else "")
            logger.info("%s: %s %s, %.1fs elapsed%s%s", self.label, completed,
                        self.unit, elapsed, remaining, f", {detail}" if detail else "")
            self.last_report = now

    def finish(self):
        logger.info("%s: finished (%d %s, %.1fs)", self.label, self.count,
                    self.unit, monotonic() - self.started)


def iter_progress(items, label, *, total=None, unit="items"):
    """Keep iteration ordered and report only work the consumer has completed."""
    progress = Progress(label, total, unit)
    for count, item in enumerate(items, 1):
        yield item
        progress.update(count)
    progress.finish()
