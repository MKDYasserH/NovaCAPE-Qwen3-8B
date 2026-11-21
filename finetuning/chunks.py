"""Find and read the embedding chunks that data/load_data.py downloads."""
import zlib
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq

from schemas.training import ChunkId, PairArrays


def hash_ids(ids: list[str]) -> np.ndarray:
    """String ids -> int64, so they can be compared on the GPU. A rare crc32 collision only masks one extra negative."""
    return np.fromiter((zlib.crc32(value.encode()) for value in ids), dtype=np.int64, count=len(ids))


class ChunkStore:
    """Pairs live in <data_dir>/qa/<shard>/chunk_NNNN_*, their hard negatives in <data_dir>/neg/<shard>/."""

    def __init__(self, data_dir: str):
        self.qa_dir = Path(data_dir) / "qa"
        self.negatives_dir = Path(data_dir) / "neg"

    def list_chunks(self) -> list[ChunkId]:
        # Each chunk has _q, _a and _meta files; matching only _q lists every chunk once.
        return [self._chunk_id(path) for path in sorted(self.qa_dir.glob("shard_*/chunk_*_q.npy"))]

    def has_hard_negatives(self, chunk: ChunkId) -> bool:
        return self._file(self.negatives_dir, chunk, "n.npy").exists()

    def read(self, chunk: ChunkId, with_hard_negatives: bool, *, mmap_mode: str | None = None) -> PairArrays:
        meta = pq.read_table(self._file(self.qa_dir, chunk, "meta.parquet"), columns=["doc_id"])
        pairs = PairArrays(
            states=np.load(self._file(self.qa_dir, chunk, "q.npy"), mmap_mode=mmap_mode),
            actions=np.load(self._file(self.qa_dir, chunk, "a.npy"), mmap_mode=mmap_mode),
            doc_ids=hash_ids(meta.column("doc_id").to_pylist()),
        )
        if with_hard_negatives:
            pairs.hard_negatives = np.load(self._file(self.negatives_dir, chunk, "n.npy"), mmap_mode=mmap_mode)
            pairs.negative_counts = np.load(self._file(self.negatives_dir, chunk, "ncnt.npy"), mmap_mode=mmap_mode)
        return pairs

    @staticmethod
    def _file(root: Path, chunk: ChunkId, suffix: str) -> Path:
        return root / chunk.shard / f"chunk_{chunk.index:04d}_{suffix}"

    @staticmethod
    def _chunk_id(path: Path) -> ChunkId:
        # shard_05/chunk_0073_q.npy -> shard_05, 73
        return ChunkId(shard=path.parent.name, index=int(path.name.split("_")[1]))
