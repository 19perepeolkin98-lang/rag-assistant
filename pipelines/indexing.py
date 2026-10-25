"""Пайплайн индексации: Markdown → чанки → BGE-эмбеддинги → FAISS.

Косинусное сходство реализовано через faiss.IndexFlatIP: все векторы
(и чанков, и запросов) L2-нормализованы, поэтому скалярное произведение
в точности равняется косинусу. IndexFlatIP даёт точный (не приближённый)
поиск — для прототипных корпусов это быстрее и честнее ANN.

Результат атомарно сохраняется в storage/:
  index.faiss — векторы;  chunks.json — тексты чанков (порядок = id векторов);
  corpus.md   — копия корпуса;  meta.json — паспорт индекса (пишется последней).
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

import faiss
import numpy as np

import config
from pipelines.chunking import chunk_markdown

# Совместимость формата хранилища (мета) — на случай будущих изменений.
STORAGE_VERSION = 2

ProgressFn = Callable[[int, int], None]  # callback(done, total)


def model_name_of(model) -> str:
    """Имя SentenceTransformer-модели независимо от версии библиотеки."""
    name = getattr(model, "name_or_path", None)
    if name:
        return str(name)
    card = getattr(model, "model_card_data", None)
    if card is not None:
        # sentence-transformers >= 3.x: имя лежит в карточке модели
        name = getattr(card, "base_model", None) or getattr(card, "model_name", None)
        if name:
            return str(name)
    return "unknown"


@dataclass
class IndexingStats:
    num_chunks: int
    num_sections: int
    dim: int
    seconds: float
    corpus_chars: int
    model_name: str


class IndexingPipeline:
    """Один вызов run() = полная перестройка индекса по переданному тексту."""

    def __init__(
        self,
        model,                       # SentenceTransformer (BGE)
        storage_dir: Path = config.STORAGE_DIR,
        batch_size: int = config.BATCH_SIZE,
    ):
        self.model = model
        self.storage_dir = Path(storage_dir)
        self.batch_size = max(1, batch_size)

    def run(
        self,
        md_text: str,
        chunk_size: int,
        chunk_overlap: int,
        progress: ProgressFn | None = None,
    ) -> IndexingStats:
        started = time.perf_counter()

        # 1. Чанкинг
        chunks = chunk_markdown(md_text, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            raise ValueError("Не удалось выделить чанки: документ пуст.")

        # 2. Эмбеддинги (батчами, с колбэком прогресса для UI)
        vectors = self._embed([c.text for c in chunks], progress)

        # 3. FAISS-индекс: норм. векторы + Inner Product = косинус
        index = faiss.IndexFlatIP(vectors.shape[1])
        index.add(vectors)

        # 4. Персист
        num_sections = len({tuple(c.heading_path) for c in chunks})
        self._save(index, chunks, md_text, chunk_size, chunk_overlap)

        return IndexingStats(
            num_chunks=len(chunks),
            num_sections=num_sections,
            dim=int(vectors.shape[1]),
            seconds=time.perf_counter() - started,
            corpus_chars=len(md_text),
            model_name=model_name_of(self.model),
        )

    # ------------------------------------------------------------------

    def _embed(self, texts: list[str], progress: ProgressFn | None) -> np.ndarray:
        out = []
        total = len(texts)
        done = 0
        for start in range(0, total, self.batch_size):
            batch = texts[start:start + self.batch_size]
            emb = self.model.encode(
                batch,
                normalize_embeddings=True,      # L2 → IP == косинус
                convert_to_numpy=True,
                show_progress_bar=False,
            )
            out.append(np.asarray(emb, dtype="float32"))
            done += len(batch)
            if progress:
                progress(done, total)
        return np.vstack(out)

    def _save(self, index, chunks, md_text, chunk_size, chunk_overlap) -> None:
        self.storage_dir.mkdir(parents=True, exist_ok=True)

        meta = {
            "storage_version": STORAGE_VERSION,
            "model": model_name_of(self.model),
            "dim": int(index.d),
            "chunk_size": chunk_size,
            "chunk_overlap": chunk_overlap,
            "num_chunks": len(chunks),
            "corpus_chars": len(md_text),
            "corpus_sha256": hashlib.sha256(md_text.encode("utf-8")).hexdigest()[:16],
            "indexed_at": datetime.now().isoformat(timespec="seconds"),
        }

        def atomic_write(path: Path, write_fn) -> None:
            tmp = path.with_suffix(path.suffix + ".tmp")
            write_fn(tmp)
            os.replace(tmp, path)   # атомарно на одном диске

        atomic_write(self.storage_dir / config.INDEX_FILE,
                     lambda p: faiss.write_index(index, str(p)))
        atomic_write(
            self.storage_dir / config.CHUNKS_FILE,
            lambda p: p.write_text(
                json.dumps([asdict(c) for c in chunks], ensure_ascii=False, indent=1),
                encoding="utf-8"),
        )
        atomic_write(self.storage_dir / config.CORPUS_FILE,
                     lambda p: p.write_text(md_text, encoding="utf-8"))
        # meta пишется последней — это «коммит»: пока её нет, индекс невалиден
        atomic_write(
            self.storage_dir / config.META_FILE,
            lambda p: p.write_text(
                json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8"),
        )
