# -*- coding: utf-8 -*-
"""Headless-смоук-тест обоих пайплайнов без Streamlit-интерфейса.

Запуск:
    python scripts/smoke_test.py [путь/к/корпусу.md] [запрос]

По умолчанию: data/sample_corpus.md и русский тестовый запрос.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import config  # noqa: E402
from config import query_instruction  # noqa: E402
from pipelines.indexing import IndexingPipeline  # noqa: E402
from pipelines.search import SearchPipeline  # noqa: E402

corpus_path = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("data/sample_corpus.md")
query = sys.argv[2] if len(sys.argv) > 2 else "сломался лифт в жилом доме"

md_text = corpus_path.read_text(encoding="utf-8")
print(f"Корпус: {corpus_path} ({len(md_text):,} символов)")

t0 = time.perf_counter()
from sentence_transformers import SentenceTransformer  # noqa: E402

print(f"Загружаю модель {config.EMBEDDING_MODEL} …")
model = SentenceTransformer(config.EMBEDDING_MODEL)
dim_fn = getattr(model, "get_embedding_dimension",
                 getattr(model, "get_sentence_embedding_dimension", None))
print(f"Модель загружена за {time.perf_counter() - t0:.1f} с, dim = {dim_fn()}")

# --- Пайплайн 1: индексация ---------------------------------------------------
print("\n=== ПАЙПЛАЙН ИНДЕКСАЦИИ ===")
stats = IndexingPipeline(model).run(
    md_text,
    chunk_size=config.CHUNK_SIZE,
    chunk_overlap=config.CHUNK_OVERLAP,
    progress=lambda done, total: None,
)
print(
    f"Чанков: {stats.num_chunks} | секций: {stats.num_sections} | "
    f"dim: {stats.dim} | время: {stats.seconds:.2f} с"
)
print(f"Сохранено в: {config.STORAGE_DIR}")

# --- Пайплайн 2: поиск ----------------------------------------------------------
print("\n=== ПАЙПЛАЙН ПОИСКА ===")
sp = SearchPipeline.load(model)
print(f"Индекс: {sp.num_chunks} чанков, построен {sp.indexed_at}, модель {sp.index_model}")

hits = sp.search(query, model, k=5, instruction=query_instruction(config.EMBEDDING_MODEL))
print(f"\nЗапрос: «{query}»")
for h in hits:
    preview = h.text.replace("\n", " ")[:100]
    print(f"  #{h.rank}  cos={h.score:+.3f}  {preview}…")

if not hits:
    sys.exit("Поиск не вернул результатов — что-то сломано.")
print("\nСмоук-тест пройден")
