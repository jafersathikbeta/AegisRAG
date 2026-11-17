"""
Minimal FAISS-compatible IndexFlatIP + helpers for environments where
faiss-cpu has no wheel (e.g. CPython 3.15 on Windows).

When real `faiss` is importable, this module re-exports it.
"""

from __future__ import annotations

try:
    import faiss as _faiss  # type: ignore

    IndexFlatIP = _faiss.IndexFlatIP
    normalize_L2 = _faiss.normalize_L2
    write_index = _faiss.write_index
    read_index = _faiss.read_index
    USING_NATIVE_FAISS = True
except ImportError:
    import numpy as np

    USING_NATIVE_FAISS = False

    def normalize_L2(x: np.ndarray) -> None:
        norms = np.linalg.norm(x, axis=1, keepdims=True)
        norms = np.maximum(norms, 1e-12)
        x /= norms

    class IndexFlatIP:
        """Inner-product flat index (numpy backend)."""

        def __init__(self, d: int) -> None:
            self.d = d
            self._vectors: np.ndarray | None = None

        @property
        def ntotal(self) -> int:
            return 0 if self._vectors is None else int(self._vectors.shape[0])

        def add(self, x: np.ndarray) -> None:
            x = np.asarray(x, dtype=np.float32)
            if self._vectors is None:
                self._vectors = x.copy()
            else:
                self._vectors = np.vstack([self._vectors, x])

        def search(self, x: np.ndarray, k: int):
            x = np.asarray(x, dtype=np.float32)
            if self._vectors is None or self.ntotal == 0:
                scores = np.full((x.shape[0], k), -1e9, dtype=np.float32)
                indices = np.full((x.shape[0], k), -1, dtype=np.int64)
                return scores, indices
            sims = x @ self._vectors.T  # (nq, ntotal)
            k = min(k, self.ntotal)
            # argpartition for top-k
            idx = np.argpartition(-sims, kth=k - 1, axis=1)[:, :k]
            row = np.arange(x.shape[0])[:, None]
            top_scores = sims[row, idx]
            order = np.argsort(-top_scores, axis=1)
            indices = idx[row, order].astype(np.int64)
            scores = top_scores[row, order].astype(np.float32)
            return scores, indices

    def write_index(index: IndexFlatIP, path: str) -> None:
        import numpy as np

        vectors = index._vectors if index._vectors is not None else np.zeros((0, index.d), dtype=np.float32)
        np.savez(path, d=np.array([index.d], dtype=np.int32), vectors=vectors)

    def read_index(path: str) -> IndexFlatIP:
        import numpy as np

        data = np.load(path)
        idx = IndexFlatIP(int(data["d"][0]))
        vectors = data["vectors"]
        if vectors.shape[0] > 0:
            idx.add(vectors.astype(np.float32))
        return idx
