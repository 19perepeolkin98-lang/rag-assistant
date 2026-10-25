"""Чат-пайплайн: полноценный диалоговый ассистент поверх GigaChat.

Три режима:
  rag        — ответы по внутренней базе (RAG); классификатор router.py
               решает, относится ли вопрос к базе, и результат сообщается
               пользователю; если нет — ассистент отвечает сам;
  reasoning  — самостоятельные структурированные рассуждения LLM (без базы);
  chat       — лёгкий свободный диалог (без базы).

Все режимы знают базовую персону («ассистент по обращениям граждан»), поэтому
«кто ты?» работает везде. История диалога передаётся в каждый запрос
(последние реплики), что даёт уточняющие вопросы вида «а в Татарстане?».
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

from config import EMBEDDING_MODEL, query_instruction
from pipelines.router import route_query
from pipelines.search import SearchHit, SearchPipeline

MODE_RAG = "rag"
MODE_REASONING = "reasoning"
MODE_CHAT = "chat"
MODES = (MODE_RAG, MODE_REASONING, MODE_CHAT)

# RAG-промпт для случая «база не подключена» (вопрос не про базу / пусто)
MODE_RAG_NODOCS = MODE_RAG + "_nodocs"

BASE_IDENTITY = (
    "Ты — ассистент Народного фронта по обращениям граждан. "
    "Ты умеешь искать по внутренней базе обращений и справочника, "
    "отвечать на вопросы по ней, рассуждать на общие темы и просто общаться."
)

MODE_SYSTEM_PROMPTS = {
    MODE_RAG: BASE_IDENTITY + """

Сейчас тебе дан вопрос пользователя и пронумерованные фрагменты внутренней базы
(обращения граждан и справочник). Правила:
1. Отвечай ТОЛЬКО на основе фрагментов, ничего не выдумывай.
2. Ссылайся на использованные фрагменты в квадратных скобках: [1], [2].
3. Если во фрагментах нет ответа — прямо скажи об этом.
4. Отвечай кратко, по делу, на русском языке.""",
    MODE_RAG_NODOCS: BASE_IDENTITY + """

Классификатор решил, что вопрос пользователя НЕ относится к внутренней базе,
поэтому фрагменты не передаются. Ответь сам, как ассистент: коротко и по делу.
Если спрашивают о твоих возможностях — расскажи: поиск по базе обращений,
ответы по справочнику, режим рассуждений, свободное общение.""",
    MODE_REASONING: BASE_IDENTITY + """

Режим рассуждений: внутренняя база документов недоступна, отвечай своими
знаниями. Структурируй мысль: сначала краткий вывод, затем рассуждение по
шагам; где уместно — списки и итог. Пиши на русском, без воды.""",
    MODE_CHAT: BASE_IDENTITY + """

Режим свободного общения: беседуй легко и дружелюбно, короткими репликами,
без канцелярита и лишних списков, как обычный собеседник. На русском языке.""",
}

SUGGESTION_PROMPT = """Ниже — последний обмен репликами с ассистентом по обращениям граждан.

Вопрос пользователя: {query}

Ответ ассистента (сокращён): {answer}

Предложи ровно 3 коротких (до 8 слов) варианта, что пользователю спросить
дальше, чтобы развить диалог, учитывая тему и режим. Верни ТОЛЬКО
JSON-массив из трёх строк без пояснений, например: ["...", "...", "..."]"""


@dataclass
class ChatReply:
    content: str
    mode: str
    note: str = ""
    used_hits: list[SearchHit] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)


class ChatPipeline:
    def __init__(
        self,
        client,                       # gigachat.GigaChat
        search_pipeline: SearchPipeline | None,
        embedder,
        model_name: str,
        temperature: float,
        max_tokens: int,
        context_hits: int,
        score_threshold: float,
        history_messages: int = 8,
        suggest_temperature: float = 0.7,
    ):
        self.client = client
        self.search_pipeline = search_pipeline
        self.embedder = embedder
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.context_hits = context_hits
        self.score_threshold = score_threshold
        self.history_messages = history_messages
        self.suggest_temperature = suggest_temperature

    # ------------------------------------------------------------------

    def ask(self, query: str, mode: str, history: list[dict]) -> ChatReply:
        """Один ход диалога: вопрос + режим + история → ответ ассистента."""
        if mode not in MODES:
            raise ValueError(f"Неизвестный режим чата: {mode}")

        instruction = query_instruction(EMBEDDING_MODEL)
        rag_hits: list[SearchHit] = []
        note = ""

        # 1. Выбираем системный промпт и (для RAG) тянем фрагменты базы
        if mode == MODE_REASONING:
            system_key, note = MODE_REASONING, "🧠 Режим рассуждений — отвечаю своими знаниями, без базы."
        elif mode == MODE_CHAT:
            system_key, note = MODE_CHAT, "💬 Свободный чат — без базы документов."
        elif self.search_pipeline is None:
            system_key, note = MODE_RAG_NODOCS, "⚠️ База ещё не проиндексирована — отвечаю без документов."
        else:
            decision = route_query(query, self.search_pipeline, self.embedder,
                                   instruction, self.score_threshold)
            if decision.use_rag:
                rag_hits = [
                    h for h in self.search_pipeline.search(
                        query, self.embedder, k=self.context_hits,
                        instruction=instruction,
                    ) if h.score >= self.score_threshold
                ][: self.context_hits]
            if rag_hits:
                system_key = MODE_RAG
                note = decision.note + f" Опора на {len(rag_hits)} фрагментов базы."
            else:
                system_key = MODE_RAG_NODOCS
                if decision.use_rag:
                    note = "🧭 Классификатор пустил вопрос в базу, но релевантных " \
                           "фрагментов нет (все ниже порога) — отвечаю без документов."
                else:
                    note = decision.note

        # 2. Собираем сообщения: система + история + вопрос (с контекстом, если есть)
        if rag_hits:
            context = "\n\n".join(
                f"[{h.rank}] (сходство {h.score:.2f} · "
                f"{' > '.join(h.heading_path) if h.heading_path else 'без раздела'} · "
                f"строки {h.start_line}–{h.end_line})\n{h.text}"
                for h in rag_hits
            )
            user_block = f"Фрагменты базы:\n\n{context}\n\nВопрос: {query}"
        else:
            user_block = query

        messages = (
            [{"role": "system", "content": MODE_SYSTEM_PROMPTS[system_key]}]
            + self._trim_history(history)
            + [{"role": "user", "content": user_block}]
        )

        from gigachat.models import Chat, Messages

        response = self.client.chat(Chat(
            model=self.model_name,
            messages=[Messages(**m) for m in messages],
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        ))
        answer = (response.choices[0].message.content or "").strip()

        return ChatReply(
            content=answer,
            mode=mode,
            note=note,
            used_hits=rag_hits,
            suggestions=self._suggest(query, answer),
        )

    # ------------------------------------------------------------------

    def _trim_history(self, history: list[dict]) -> list[dict]:
        """Последние N реплик, каждая урезана до разумной длины."""
        msgs = [
            {"role": m["role"], "content": m["content"][:1500]}
            for m in history
            if m.get("role") in ("user", "assistant") and m.get("content")
        ]
        return msgs[-self.history_messages:]

    def _suggest(self, query: str, answer: str) -> list[str]:
        """Три коротких продолжения диалога (кнопки «спросить ещё»).

        LLM иногда отвечает текстом вместо JSON — на такой случай даём
        одну запасную попытку с более жёсткой инструкцией.
        """
        attempts = (
            SUGGESTION_PROMPT.format(query=query[:500], answer=answer[:700]),
            "Верни СТРОГО один JSON-массив из трёх коротких (до 8 слов) строк — "
            "варианты следующего вопроса пользователя. Без markdown, без "
            "пояснений. Пример формата: [\"строка1\", \"строка2\", \"строка3\"]",
        )
        try:
            from gigachat.models import Chat, Messages

            for prompt in attempts:
                response = self.client.chat(Chat(
                    model=self.model_name,
                    messages=[Messages(role="user", content=prompt)],
                    temperature=self.suggest_temperature,
                    max_tokens=200,
                ))
                raw = (response.choices[0].message.content or "").strip()
                m = re.search(r"\[.*\]", raw, flags=re.S)
                if not m:
                    continue
                try:
                    items = [str(s).strip() for s in json.loads(m.group(0)) if str(s).strip()]
                except json.JSONDecodeError:
                    continue
                if items:
                    return items[:3]
            return []
        except Exception:  # noqa: BLE001 — подсказки необязательны, без них просто живём
            return []
