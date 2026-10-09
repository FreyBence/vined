"""Console progress bars with stage messages only between active bars."""

import logging
from tqdm import tqdm


logger = logging.getLogger("vined.progress")
_active_bars = 0


class _StageHandler(logging.StreamHandler):
    def __init__(self):
        super().__init__()
        self.pending = []

    def emit(self, record):
        if _active_bars:
            self.pending.append(record)
        else:
            super().emit(record)

    def flush_pending(self):
        pending, self.pending = self.pending, []
        for record in pending:
            super().emit(record)


def configure_progress():
    """Enable stderr bars and timestamped messages between pipeline stages."""
    if not logger.handlers:
        handler = _StageHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
        logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False


class Progress:
    """An in-place bar; use as a context manager to close it on failures."""

    def __init__(self, label, total=None, unit="items", *, disable=False):
        global _active_bars
        self.count = 0
        self.closed = False
        self.bar = tqdm(total=total, desc=label, unit=unit, dynamic_ncols=True,
                        mininterval=0.5, miniters=0,
                        disable=disable or not logger.isEnabledFor(logging.INFO))
        self.active = not self.bar.disable
        if self.active:
            _active_bars += 1

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.finish()

    def update(self, count, detail=""):
        if detail:
            self.bar.set_postfix_str(detail, refresh=False)
        self.bar.update(count - self.count)
        self.count = count

    def finish(self):
        global _active_bars
        if self.closed:
            return
        self.closed = True
        self.bar.close()
        if self.active:
            _active_bars -= 1
        if not _active_bars:
            for handler in logger.handlers:
                if isinstance(handler, _StageHandler):
                    handler.flush_pending()


def iter_progress(items, label, *, total=None, unit="items"):
    """Keep iteration ordered and report only work the consumer has completed."""
    with Progress(label, total, unit) as progress:
        for count, item in enumerate(items, 1):
            yield item
            progress.update(count)
