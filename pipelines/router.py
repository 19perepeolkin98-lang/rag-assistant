"""Диспетчер запросов: решает, нужен ли поиск по внутренней базе (RAG).

Два уровня классификации:
1. Служебные вопросы о самом ассистенте («кто ты?», «что ты умеешь?») —
   распознаются шаблонами, база не нужна.
2. Всё остальное — оцениваем уверенность ретривера: косинус самого близкого
   чанка. Выше порога — вопрос «про нашу документацию», ниже — общение
   или общие знания, RAG не подключаем.

Такой классификатор не требует ни дообучения, ни отдельного LLM-вызова:
порог уже откалиброван на корпусе (релевантные вопросы дают 0.55–0.70,
нерелевантные — ~0.30).
"""
from __future__ import annotations

import re
from dataclasses import dataclass

# Вопросы о самом ассистенте / этикетные реплики
META_PATTERNS = [
    r"\bкто ты\b", r"\bты кто\b", r"что ты (такое|умеешь|можешь)",
    r"твои (функции|возможности)",
    r"как тобой (пользоваться|пользоваться\?)",
    r"\bпривет\b", r"здравствуй", r"\bспасибо\b", r"благодарю",
    r"как (у тебя )?дела", r"\bхай\b",
]
_META_RE = [re.compile(p) for p in META_PATTERNS]


@dataclass
class RouteDecision:
    use_rag: bool
    note: str            # человеческое объяснение для UI
    top_score: float | None = None


def is_meta_question(query: str) -> bool:
    q = query.lower().strip()
    return any(rx.search(q) for rx in _META_RE)


def route_query(
    query: str,
    search_pipeline: SearchPipeline,
    embedder,
    instruction: str,
    threshold: float,
) -> RouteDecision:
    """Решает, подключать ли RAG. Сам поиск дешёвый (~миллисекунды)."""
    if is_meta_question(query):
        return RouteDecision(
            use_rag=False,
            note="🧭 Классификатор: вопрос о самом ассистенте — поиск по базе не нужен.",
        )

    hits = search_pipeline.search(query, embedder, k=1, instruction=instruction)
    if not hits:
        return RouteDecision(
            use_rag=False,
            note="🧭 Классификатор: база пуста — отвечаю без документов.",
        )

    top = hits[0].score
    if top >= threshold:
        return RouteDecision(
            use_rag=True,
            note=f"🧭 Классификатор: вопрос похож на тему внутренней документации "
                 f"(cos = {top:.2f}) → 🔎 ищу по базе обращений.",
            top_score=top,
        )
    return RouteDecision(
        use_rag=False,
        note=f"🧭 Классификатор: вопрос не похож на темы базы "
             f"(лучший cos = {top:.2f}) — отвечаю без документов.",
        top_score=top,
    )
