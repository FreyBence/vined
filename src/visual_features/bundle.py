"""Bounded common-pass publication of the three E2 representations."""

from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from queue import Queue, Empty, Full
from threading import Event
from types import SimpleNamespace

from .artifacts import write_features
from .encoder import EncodedObservation, iter_encoded_observations


def write_feature_bundle(observations, encoder, outputs, *, batch_size=64, workers=1):
    """Write separate integrity-checked A/B/G archives from one inference stream.

    Each queue holds at most two observations. Any writer failure cancels the
    producer; no bundle completion is returned unless all three readers pass.
    Already published individual artifacts retain their normal immutable status.
    """
    if set(outputs) != {"A", "B", "G"}:
        raise ValueError("Exactly A/B/G output paths are required")
    from pathlib import Path
    if any(Path(p).exists() for p in outputs.values()):
        raise FileExistsError("Bundle destinations must all be new")
    queues = {arm: Queue(maxsize=2) for arm in outputs}
    stopped = Event()
    sentinel = object()

    def consume(arm):
        try:
            while True:
                try:
                    item = queues[arm].get(timeout=0.2)
                except Empty:
                    if stopped.is_set():
                        raise RuntimeError("Common-pass extraction interrupted")
                    continue
                if item is sentinel:
                    return
                yield item
        finally:
            stopped.set()

    def send(arm, item):
        while True:
            if stopped.is_set():
                raise RuntimeError("An E2 feature writer stopped before bundle completion")
            try:
                queues[arm].put(item, timeout=0.2)
                return
            except Full:
                pass

    def writer(arm):
        try:
            proxy = SimpleNamespace(provenance=encoder.representation_provenance(arm))
            return write_features(observations, proxy, outputs[arm], batch_size=batch_size,
                                  workers=workers, _encoded=consume(arm))
        except BaseException:
            stopped.set()
            raise

    with ThreadPoolExecutor(max_workers=3, thread_name_prefix="e2-writer") as pool:
        futures = {arm: pool.submit(writer, arm) for arm in outputs}
        try:
            for item in iter_encoded_observations(observations, encoder, batch_size=batch_size, workers=workers):
                for arm, section in (("A", slice(0, 768)), ("B", slice(768, 4864)), ("G", slice(4864, 8960))):
                    derived = (replace(item, feature=item.feature[section].copy())
                               if isinstance(item, EncodedObservation) and item.feature is not None else item)
                    send(arm, derived)
            # Do not treat normal exhaustion of one consumer as producer failure.
            for queue in queues.values():
                queue.put(sentinel)
            return {arm: future.result() for arm, future in futures.items()}
        finally:
            stopped.set()
