# -*- coding: utf-8 -*-
"""Живая проверка GigaChat API + калибровка порога релевантности.

1) Ищет осмысленные и абсурдный запросы — видно распределение косинусов.
2) Дергает GigaChat минимальным промптом — проверяет ключ из .env.
3) Полный RAG-прогон: поиск → контекст → ответ.

Запуск: python scripts/test_gigachat.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# GigaChat (сберовские хосты) не должен идти через системный прокси/VPN:
# зарубежный выход режется geo-блокировкой Сбера → SSL EOF.
os.environ.setdefault(
    "NO_PROXY",
    "sberbank.ru,.sberbank.ru,gigachat.devices.sberbank.ru,ngw.devices.sberbank.ru",
)

import config  # noqa: E402
from config import query_instruction  # noqa: E402
from pipelines.generation import GenerationPipeline  # noqa: E402
from pipelines.search import SearchPipeline  # noqa: E402
from sentence_transformers import SentenceTransformer  # noqa: E402

model = SentenceTransformer(config.EMBEDDING_MODEL)
sp = SearchPipeline.load(model)

for q in ["куда жаловаться на управляющую компанию",
          "какие проблемы с лифтами в Татарстане",
          "рецепт борща из свёклы"]:
    hits = sp.search(q, model, k=5, instruction=query_instruction(config.EMBEDDING_MODEL))
    scores = ", ".join(f"{h.score:+.3f}" for h in hits)
    print(f"«{q}»\n  скоры: {scores}\n  top: {hits[0].text.splitlines()[-1][:70]}")

print("\n--- GigaChat API ---")
if not config.GIGACHAT_CREDENTIALS:
    sys.exit("GIGACHAT_CREDENTIALS не задан в .env")

from gigachat import GigaChat  # noqa: E402
from gigachat.models import Chat, Messages  # noqa: E402

with GigaChat(
    credentials=config.GIGACHAT_CREDENTIALS,
    scope=config.GIGACHAT_SCOPE,
    model=config.GIGACHAT_MODEL,
    verify_ssl_certs=config.GIGACHAT_VERIFY_SSL,
    timeout=config.GIGACHAT_TIMEOUT,
) as client:
    models = client.get_models()
    ids = [getattr(m, "id_", getattr(m, "id", None)) for m in models.data]
    print("Доступные модели:", ids)
    resp = client.chat(Chat(
        model=config.GIGACHAT_MODEL,
        messages=[Messages(role="user", content="Скажи одно слово: работает")],
        max_tokens=16,
    ))
    print("GigaChat ответил:", resp.choices[0].message.content)

    # Полный прогон генерации по реальным результатам поиска
    q = "куда жаловаться на управляющую компанию"
    hits = sp.search(q, model, k=5, instruction=query_instruction(config.EMBEDDING_MODEL))
    gen = GenerationPipeline(
        score_threshold=config.RELEVANCE_THRESHOLD,
        max_hits=config.GENERATION_CONTEXT_HITS,
        temperature=config.GENERATION_TEMPERATURE,
        max_tokens=config.GENERATION_MAX_TOKENS,
        model_name=config.GIGACHAT_MODEL,
    )
    result = gen.run(q, hits, client)
    print("\n--- Генерация по RAG-контексту ---")
    print("no_relevant:", result.no_relevant, "| использовано чанков:", len(result.used_hits))
    print(result.answer)
