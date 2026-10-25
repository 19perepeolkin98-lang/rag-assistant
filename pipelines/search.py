"""Пайплайн поиска: запрос → BGE-эмбеддинг → косинусное сходство в FAISS → top-k.

Полностью отделён от пайплайна индексации: знает только про готовый индекс
в storage/ и умеет только искать. Косинус == Inner Product по L2-нормализованным
векторам (см. pipelines/indexing.py).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import faiss

import config
from pipelines.indexing import model_name_of


@dataclass
class SearchHit:
    rank: int
    score: float            # косинусное сходство, [-1..1]
    text: str
    heading_path: list[str]
    start_line: int
    end_line: int


class IndexNotFoundError(Exception):
    """Индекса ещё нет — нужно сначала запустить пайплайн индексации."""


class ModelMismatchError(Exception):
    """Индекс построен другой моделью эмбеддингов — нужен реиндекс."""


class SearchPipeline:
    def __init__(self, index: "faiss.Index", chunks: list[dict], meta: dict):
        self.index = index
        self.chunks = chunks
        self.meta = meta

    # ------------------------------------------------------------------

    @classmethod
    def load(cls, model, storage_dir: Path = config.STORAGE_DIR) -> "SearchPipeline":
        """Загружает индекс с диска и проверяет совместимость с моделью."""
        storage = Path(storage_dir)
        index_path = storage / config.INDEX_FILE
        meta_path = storage / config.META_FILE
        chunks_path = storage / config.CHUNKS_FILE

        if not (index_path.exists() and meta_path.exists() and chunks_path.exists()):
            raise IndexNotFoundError(
                "Индекс не найден. Сначала загрузите Markdown-корпус "
                "на вкладке «Индексация»."
            )

        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
        index = faiss.read_index(str(index_path))

        active = model_name_of(model)
        if meta.get("model") != active:
            raise ModelMismatchError(
                f"Индекс построен моделью «{meta.get('model')}», а сейчас активна "
                f"«{active}». Измените EMBEDDING_MODEL в .env обратно или "
                f"переиндексируйте корпус на вкладке «Индексация»."
            )
        # get_embedding_dimension — новое имя метода (ST >= 3.x переименовала)
        dim_fn = getattr(model, "get_embedding_dimension", None) \
            or getattr(model, "get_sentence_embedding_dimension", None)
        dim = dim_fn() if dim_fn else None
        if dim is not None and int(dim) != index.d:
            raise ModelMismatchError(
                f"Размерность индекса ({index.d}) не совпадает с моделью ({dim}). "
                "Переиндексируйте корпус."
            )
        return cls(index, chunks, meta)

    # ------------------------------------------------------------------

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    @property
    def indexed_at(self) -> str:
        return self.meta.get("indexed_at", "—")

    @property
    def index_model(self) -> str:
        return self.meta.get("model", "—")

    def search(self, query: str, model, k: int = 5, instruction: str = "") -> list[SearchHit]:
        """Top-k чанков по косинусному сходству с запросом."""
        query = query.strip()
        if not query:
            return []

        emb = model.encode(
            [instruction + query] if instruction else [query],
            normalize_embeddings=True,
            convert_to_numpy=True,
            show_progress_bar=False,
        ).astype("float32")

        k = max(1, min(k, self.num_chunks))
        scores, ids = self.index.search(emb, k)

        hits: list[SearchHit] = []
        for rank, (score, idx) in enumerate(zip(scores[0], ids[0]), start=1):
            if idx < 0:  # faiss паддинг -1, если k > ntotal
                continue
            chunk = self.chunks[int(idx)]
            hits.append(
                SearchHit(
                    rank=rank,
                    score=float(score),
                    text=chunk["text"],
                    heading_path=chunk.get("heading_path", []),
                    start_line=chunk.get("start_line", 0),
                    end_line=chunk.get("end_line", 0),
                )
            )
        return hits
