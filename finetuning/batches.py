"""Turns a list of chunks into shuffled training batches while loading from disk in the background.

reader thread: loads `chunks_per_pool` chunks at once and joins them into one pool.
dealer thread: shuffles each pool `pool_repeats` times and cuts it into batches.
Mixing many chunks in a pool gives each batch varied negatives without loading the whole dataset.
"""
import queue
import threading
from collections.abc import Callable, Iterator
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from finetuning.chunks import ChunkStore
from schemas.training import BatchStreamSettings, ChunkId, PairArrays

END = object()  # put on a queue after the last item
WAKE_UP_S = 1.0  # blocked threads wake up this often to check whether they should stop


class BatchStream:
    """Use as `with BatchStream(...) as batches: for batch in batches:`; leaving the block stops the threads."""

    def __init__(self, store: ChunkStore, settings: BatchStreamSettings):
        self.store = store
        self.settings = settings
        self.stop_event = threading.Event()
        self.error: BaseException | None = None
        # maxsize keeps memory bounded: at most one finished pool waits while the dealer works.
        self.pools: queue.Queue = queue.Queue(maxsize=1)
        self.batches: queue.Queue = queue.Queue(maxsize=settings.prefetch_batches)

    def __enter__(self) -> "BatchStream":
        return self

    def __exit__(self, *exc_info: object) -> None:
        self.stop_event.set()

    def __iter__(self) -> Iterator[PairArrays]:
        threading.Thread(target=self._guarded, args=(self._read_pools,), daemon=True).start()
        threading.Thread(target=self._guarded, args=(self._deal_batches,), daemon=True).start()
        while not self.stop_event.is_set():
            batch = self._get(self.batches)
            if batch is END:
                break
            yield batch
        if self.error:
            raise self.error

    def _read_pools(self) -> None:
        chunks, pool_size = self.settings.chunks, self.settings.chunks_per_pool
        with ThreadPoolExecutor(pool_size) as executor:
            for start in range(0, len(chunks), pool_size):
                if self.stop_event.is_set():
                    return
                pool = PairArrays.concat(list(executor.map(self._read_chunk, chunks[start : start + pool_size])))
                if not self._put(self.pools, pool):
                    return
        self._put(self.pools, END)

    def _deal_batches(self) -> None:
        rng = np.random.default_rng(self.settings.seed)
        batch_size = self.settings.batch_size
        while (pool := self._get(self.pools)) is not END:
            for _ in range(self.settings.pool_repeats):
                shuffled = rng.permutation(pool.num_rows)
                # Rows left over after the last full batch are skipped; each shuffle skips different ones.
                for start in range(0, pool.num_rows - batch_size + 1, batch_size):
                    # Sorted indices read memory in order, which makes the copy faster.
                    rows = np.sort(shuffled[start : start + batch_size])
                    if not self._put(self.batches, pool.take(rows)):
                        return
        self._put(self.batches, END)

    def _read_chunk(self, chunk: ChunkId) -> PairArrays:
        return self.store.read(chunk, self.settings.with_hard_negatives)

    def _put(self, target: queue.Queue, item: object) -> bool:
        """Blocks until there is room. Returns False if the stream was stopped meanwhile."""
        while not self.stop_event.is_set():
            try:
                target.put(item, timeout=WAKE_UP_S)
                return True
            except queue.Full:
                pass
        return False

    def _get(self, source: queue.Queue) -> object:
        """Blocks until an item arrives. Returns END if the stream was stopped meanwhile."""
        while not self.stop_event.is_set():
            try:
                return source.get(timeout=WAKE_UP_S)
            except queue.Empty:
                pass
        return END

    def _guarded(self, work: Callable[[], None]) -> None:
        # Without this, a crash in a thread leaves the training loop waiting forever.
        try:
            work()
        except BaseException as error:
            self.error = error
            self.stop_event.set()
