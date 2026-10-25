"""Пайплайн генерации: вопрос + найденные чанки → GigaChat → ответ с опорой.

Третий, независимый пайплайн: берёт на входе результаты пайплайна поиска,
фильтрует их по порогу релевантности (косинус), собирает промпт и вызывает
GigaChat по API. Если ни один чанк не прошёл порог — LLM не вызывается
вовсе (защита от галлюцинаций), вызывающий код получает no_relevant=True.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from pipelines.search import SearchHit

SYSTEM_PROMPT = """Ты — аналитик Народного фронта, помощник по обращениям граждан.

Тебе дан вопрос пользователя и пронумерованные фрагменты базы обращений/справочника.
Правила:
1. Отвечай ТОЛЬКО на основе фрагментов контекста, ничего не выдумывай.
2. Ссылайся на использованные фрагменты в квадратных скобках: [1], [2].
3. Если информации в контексте недостаточно — прямо скажи об этом.
4. Отвечай кратко, по делу, на русском языке."""


class LLMError(Exception):
    """Ошибка обращения к GigaChat API (сеть, авторизация, лимиты)."""


@dataclass
class GenerationResult:
    answer: str
    used_hits: list[SearchHit] = field(default_factory=list)
    skipped_low_score: int = 0
    no_relevant: bool = False   # True = ни один чанк не прошёл порог


class GenerationPipeline:
    def __init__(
        self,
        score_threshold: float,
        max_hits: int,
        temperature: float,
        max_tokens: int,
        model_name: str = "GigaChat-2",
    ):
        self.score_threshold = score_threshold
        self.max_hits = max_hits
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.model_name = model_name

    # ------------------------------------------------------------------

    def run(self, query: str, hits: list[SearchHit], client) -> GenerationResult:
        """client — инициализированный gigachat.GigaChat."""
        relevant = [h for h in hits if h.score >= self.score_threshold][: self.max_hits]
        if not relevant:
            return GenerationResult(
                answer="", skipped_low_score=len(hits), no_relevant=True
            )

        context_blocks = []
        for h in relevant:
            crumb = " > ".join(h.heading_path) if h.heading_path else "без раздела"
            context_blocks.append(
                f"[{h.rank}] (сходство {h.score:.2f} · {crumb} · строки {h.start_line}–{h.end_line})\n{h.text}"
            )
        user_prompt = (
            "Фрагменты базы:\n\n"
            + "\n\n".join(context_blocks)
            + f"\n\nВопрос: {query.strip()}"
        )

        try:
            from gigachat.models import Chat, Messages

            payload = Chat(
                model=self.model_name,
                messages=[
                    Messages(role="system", content=SYSTEM_PROMPT),
                    Messages(role="user", content=user_prompt),
                ],
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
            response = client.chat(payload)
            answer = (response.choices[0].message.content or "").strip()
        except Exception as e:  # noqa: BLE001 — любую ошибку API показываем человеческим текстом
            raise LLMError(f"GigaChat API: {e}") from e

        return GenerationResult(
            answer=answer,
            used_hits=relevant,
            skipped_low_score=len(hits) - len(relevant),
        )
