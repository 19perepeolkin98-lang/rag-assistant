"""RAG на BGE-эмбеддингах + FAISS: Streamlit-интерфейс.

Две вкладки = два независимых пайплайна:
  1. «Индексация» — загрузка одного большого Markdown-файла, чанкинг,
     эмбеддинги BGE, запись FAISS-индекса в storage/.
  2. «Поиск» — запрос → эмбеддинг → косинусное сходство → top-k чанков.
"""
import json
import re
import time

import streamlit as st

import config
from config import (
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    TOP_K_DEFAULT,
    TOP_K_MAX,
    is_english_only_model,
    query_instruction,
)
from pipelines.chat import MODE_CHAT, MODE_RAG, MODE_REASONING, ChatPipeline
from pipelines.generation import GenerationPipeline, LLMError
from pipelines.indexing import IndexingPipeline
from pipelines.search import IndexNotFoundError, ModelMismatchError, SearchPipeline

st.set_page_config(
    page_title="RAG: Markdown → чанки → FAISS",
    page_icon="📄",
    layout="wide",
)

st.title("RAG на BGE-эмбеддингах + FAISS")
st.caption(
    "Загрузите один большой Markdown-файл — пайплайн нарежет его на чанки и "
    "построит FAISS-индекс. Поиск — отдельный пайплайн по косинусному сходству."
)


# ======================
# МОДЕЛЬ ЭМБЕДДИНГОВ
# ======================
@st.cache_resource(show_spinner=False)
def load_embedding_model(model_name: str):
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_name)


with st.spinner(f"Загружаю модель эмбеддингов «{EMBEDDING_MODEL}» (при первом запуске скачается из HuggingFace)..."):
    embedder = load_embedding_model(EMBEDDING_MODEL)

INSTRUCTION = query_instruction(EMBEDDING_MODEL)
EN_ONLY = is_english_only_model(EMBEDDING_MODEL)


# ======================
# GIGACHAT (генерация)
# ======================
@st.cache_resource(show_spinner=False)
def get_gigachat():
    """Клиент GigaChat API; None, если ключ не настроен."""
    if not config.GIGACHAT_CREDENTIALS:
        return None
    from gigachat import GigaChat

    return GigaChat(
        credentials=config.GIGACHAT_CREDENTIALS,
        scope=config.GIGACHAT_SCOPE,
        model=config.GIGACHAT_MODEL,
        verify_ssl_certs=config.GIGACHAT_VERIFY_SSL,
        timeout=config.GIGACHAT_TIMEOUT,
        max_retries=3,
    )


# ======================
# СОСТОЯНИЕ ИНДЕКСА
# ======================
def read_meta() -> dict | None:
    meta_path = config.STORAGE_DIR / config.META_FILE
    if not meta_path.exists():
        return None
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


@st.cache_data(show_spinner=False)
def read_meta_cached(meta_mtime: int) -> dict | None:
    # meta_mtime — часть ключа кэша: после реиндекса мета перечитается.
    # ВАЖНО: не называть аргумент с подчёркивания — такие аргументы Streamlit
    # не хэширует, и кэш не инвалидируется.
    return read_meta()


def get_meta() -> dict | None:
    """Мета индекса; кэшируется по mtime файла, чтобы после реиндекса увидеть свежую."""
    meta_path = config.STORAGE_DIR / config.META_FILE
    mtime = meta_path.stat().st_mtime_ns if meta_path.exists() else 0
    return read_meta_cached(mtime)


@st.cache_resource(show_spinner=False)
def _load_search_pipeline(meta_mtime: int, model_name: str) -> SearchPipeline | None:
    """Загружает пайплайн поиска; ключ — mtime meta.json (после реиндекса перезагрузится)."""
    try:
        return SearchPipeline.load(embedder)
    except IndexNotFoundError:
        return None


def get_search_pipeline() -> SearchPipeline | None:
    meta_path = config.STORAGE_DIR / config.META_FILE
    mtime = meta_path.stat().st_mtime_ns if meta_path.exists() else 0
    try:
        return _load_search_pipeline(mtime, EMBEDDING_MODEL)
    except IndexNotFoundError:
        return None
    # ModelMismatchError пробрасывается наружу — UI покажет понятную ошибку


META = get_meta()

# ======================
# БОКОВАЯ ПАНЕЛЬ
# ======================
with st.sidebar:
    st.header("Состояние системы")

    st.markdown(f"**Модель эмбеддингов**\n\n`{EMBEDDING_MODEL}`")
    if EN_ONLY:
        st.warning(
            "Активная BGE-модель англоязычная. Для корпуса на русском задайте в "
            "`.env` строку `EMBEDDING_MODEL=BAAI/bge-m3` и перезапустите приложение."
        )

    st.divider()
    if config.GIGACHAT_CREDENTIALS:
        st.markdown(f"**Генерация ответов**\n\nGigaChat API · модель `{config.GIGACHAT_MODEL}`")
    else:
        st.markdown("**Генерация ответов**\n\nне настроена — добавьте `GIGACHAT_CREDENTIALS` в `.env`")

    st.divider()
    if META:
        st.success("Индекс готов")
        st.markdown(
            f"- Чанков в индексе: **{META.get('num_chunks', '—')}**\n"
            f"- Размерность: **{META.get('dim', '—')}**\n"
            f"- Построен: **{META.get('indexed_at', '—')}**\n"
            f"- Модель: `{META.get('model', '—')}`"
        )
        if META.get("model") != EMBEDDING_MODEL:
            st.error(
                "Индекс построен другой моделью! Поиск будет недоступен, пока не "
                "переиндексируете корпус текущей моделью."
            )
    else:
        st.info("Индекса ещё нет — загрузите корпус на вкладке «Индексация».")

    st.divider()
    st.caption(f"Хранилище индекса: `{config.STORAGE_DIR}`")

# ======================
# ВКЛАДКА 1: ИНДЕКСАЦИЯ
# ======================
CYRILLIC_RE = re.compile(r"[а-яёА-ЯЁ]")


def cyrillic_share(text: str) -> float:
    if not text:
        return 0.0
    letters = [ch for ch in text.lower() if ch.isalpha()]
    if not letters:
        return 0.0
    return sum(bool(CYRILLIC_RE.match(ch)) for ch in letters) / len(letters)


# ======================
# ВКЛАДКИ
# ======================
tab_chat, tab_index, tab_search = st.tabs(["💬 Чат-бот", "📥 Индексация", "🔍 Поиск"])

# ======================
# ВКЛАДКА 0: ЧАТ-БОТ
# ======================
CHAT_MODE_OPTIONS = {
    "📚 По документации (RAG)": MODE_RAG,
    "🧠 Рассуждения (LLM)": MODE_REASONING,
    "💬 Свободный чат": MODE_CHAT,
}


@st.dialog("Как пользоваться ассистентом", width="large")
def show_onboarding() -> None:
    st.markdown(
        "**1️⃣ Выберите режим** — 📚 ответы по внутренней документации (RAG), "
        "🧠 рассуждения LLM или 💬 свободный чат.\n\n"
        "**2️⃣ Напишите вопрос** в поле внизу — или нажмите готовую подсказку "
        "под последним ответом.\n\n"
        "**3️⃣ Получите ответ** — в режиме RAG со ссылками на фрагменты базы "
        "вида [1], [2] и пояснением, искала ли система по документации."
    )
    st.info(
        "В режиме «По документации» классификатор сам решает, относится ли "
        "вопрос к базе: если да — увидите «🔎 ищу по внутренней документации», "
        "если нет — ассистент ответит сам и честно об этом скажет."
    )
    if st.button("Понятно, начнём!", type="primary"):
        st.rerun()


def build_chat_pipeline() -> ChatPipeline | None:
    client = get_gigachat()
    if client is None:
        return None
    try:
        sp = get_search_pipeline()
    except ModelMismatchError:
        sp = None  # чат продолжит работать без базы
    return ChatPipeline(
        client=client,
        search_pipeline=sp,
        embedder=embedder,
        model_name=config.GIGACHAT_MODEL,
        temperature=config.GENERATION_TEMPERATURE,
        max_tokens=config.GENERATION_MAX_TOKENS,
        context_hits=config.GENERATION_CONTEXT_HITS,
        score_threshold=config.RELEVANCE_THRESHOLD,
    )


with tab_chat:
    col_title, col_help = st.columns([5, 1])
    with col_title:
        st.subheader("Ассистент по обращениям граждан")
    with col_help:
        if st.button("❓ Как пользоваться"):
            show_onboarding()

    if "chat_onboarded" not in st.session_state:
        st.session_state.chat_onboarded = True
        show_onboarding()

    chat = build_chat_pipeline()
    if chat is None:
        st.error(
            "Чат-бот недоступен: в `.env` не задан `GIGACHAT_CREDENTIALS`. "
            "Вкладки «Индексация» и «Поиск» работают и без него."
        )
    else:
        mode_label = st.radio(
            "Режим ответов",
            options=list(CHAT_MODE_OPTIONS),
            horizontal=True,
        )
        mode = CHAT_MODE_OPTIONS[mode_label]
        st.caption(
            "История диалога хранится в рамках вашей сессии браузера. "
            "Классификатор работает в режиме RAG."
        )

        history: list[dict] = st.session_state.setdefault("chat_history", [])

        # --- история диалога ------------------------------------------------
        for i, msg in enumerate(history):
            with st.chat_message(msg["role"]):
                if msg.get("note"):
                    st.caption(msg["note"])
                st.markdown(msg["content"])
                if msg.get("sources"):
                    with st.expander(f"🔎 Источники ({len(msg['sources'])})"):
                        for s in msg["sources"]:
                            st.markdown(
                                f"- `[{s['rank']}]` cos = {s['score']:.2f} · {s['crumb']}"
                            )
                # подсказки-продолжения — только у последнего ответа
                if msg["role"] == "assistant" and i == len(history) - 1 and msg.get("suggestions"):
                    scols = st.columns(len(msg["suggestions"]))
                    for sc, s in zip(scols, msg["suggestions"]):
                        if sc.button(s, key=f"sugg_{i}_{hash(s) % 10**8}"):
                            st.session_state["chat_pending"] = s

        # --- ввод: поле чата или клик по подсказке --------------------------
        pending = st.session_state.pop("chat_pending", None)
        user_text = st.chat_input("Спросите о чём угодно: обращения, ЖКХ, куда жаловаться…")
        question = user_text or pending

        if question:
            history.append({"role": "user", "content": question})
            spinner_text = {
                MODE_RAG: "🔎 Классифицирую вопрос и ищу по внутренней документации…",
                MODE_REASONING: "🧠 Думаю…",
                MODE_CHAT: "💬 Отвечаю…",
            }[mode]
            try:
                with st.spinner(spinner_text):
                    reply = chat.ask(question, mode, history[:-1])
            except Exception as e:  # noqa: BLE001 — ошибку показываем, историю не ломаем
                st.error(f"Не удалось получить ответ: {e}")
            else:
                history.append({
                    "role": "assistant",
                    "content": reply.content,
                    "note": reply.note,
                    "mode": reply.mode,
                    "suggestions": reply.suggestions,
                    "sources": [
                        {
                            "rank": h.rank,
                            "score": h.score,
                            "crumb": " > ".join(h.heading_path) if h.heading_path else "без раздела",
                        }
                        for h in reply.used_hits
                    ],
                })
            st.rerun()

with tab_index:
    st.subheader("Пайплайн индексации: Markdown → чанки → эмбеддинги → FAISS")

    uploaded = st.file_uploader(
        "Загрузите один большой Markdown-файл с корпусом",
        type=["md", "markdown", "txt"],
        help="Файл будет разрезан на чанки по заголовкам и абзацам, "
        "каждый чанк превратится в BGE-вектор и попадёт в FAISS.",
    )

    if uploaded is not None:
        raw = uploaded.read()
        md_text = raw.decode("utf-8-sig", errors="replace")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Размер", f"{len(raw) / 1024:.1f} КБ")
        col2.metric("Символов", f"{len(md_text):,}")
        col3.metric("Строк", f"{md_text.count(chr(10)) + 1:,}")
        col4.metric("Заголовков #", len(re.findall(r"^#{1,6}\s", md_text, flags=re.M)))

        with st.expander("Предпросмотр начала файла"):
            st.text(md_text[:2000] + ("…" if len(md_text) > 2000 else ""))

        st.divider()
        with st.expander("Параметры чанкинга"):
            in_size = st.number_input(
                "Размер чанка, символов", min_value=50, max_value=8000,
                value=CHUNK_SIZE, step=50,
            )
            in_overlap = st.number_input(
                "Перекрытие чанков, символов", min_value=0, max_value=2000,
                value=CHUNK_OVERLAP, step=25,
            )

        if st.button("⚙️ Запустить пайплайн индексации", type="primary"):
            if in_overlap >= in_size:
                st.error("Перекрытие должно быть строго меньше размера чанка.")
            else:
                progress_bar = st.progress(0.0, text="Чанкинг и загрузка модели…")
                try:
                    pipeline = IndexingPipeline(embedder)

                    def on_progress(done: int, total: int) -> None:
                        progress_bar.progress(
                            done / total,
                            text=f"Считаю эмбеддинги: {done} / {total} чанков…",
                        )

                    stats = pipeline.run(
                        md_text,
                        chunk_size=int(in_size),
                        chunk_overlap=int(in_overlap),
                        progress=on_progress,
                    )
                except Exception as e:  # noqa: BLE001 — показываем любую ошибку пайплайна в UI
                    progress_bar.empty()
                    st.error(f"Индексация не удалась: {e}")
                else:
                    progress_bar.progress(1.0, text="Готово!")
                    st.success("Индекс построен и сохранён в storage/. Можно переходить на вкладку «Поиск».")

                    m1, m2, m3, m4 = st.columns(4)
                    m1.metric("Чанков", stats.num_chunks)
                    m2.metric("Секций (по заголовкам)", stats.num_sections)
                    m3.metric("Размерность векторов", stats.dim)
                    m4.metric("Время индексации", f"{stats.seconds:.1f} с")

                    if EN_ONLY and cyrillic_share(md_text) > 0.3:
                        st.warning(
                            f"Корпус в основном на русском ({cyrillic_share(md_text):.0%} букв кириллицы), "
                            "а активная модель bge-*-en англоязычная — качество поиска будет низким. "
                            "Поставьте в .env: EMBEDDING_MODEL=BAAI/bge-m3 и переиндексируйте."
                        )

# ======================
# ВКЛАДКА 2: ПОИСК
# ======================
with tab_search:
    st.subheader("Пайплайн поиска: запрос → косинусное сходство → top-k")

    try:
        search_pipeline = get_search_pipeline()
    except ModelMismatchError as e:
        search_pipeline = None
        st.error(str(e))

    if search_pipeline is None:
        st.info(
            "Индекс пуст. Сначала загрузите Markdown-корпус на вкладке «Индексация» — "
            "после завершения индексации поиск станет доступен здесь."
        )
    else:
        st.caption(
            f"В индексе {search_pipeline.num_chunks} чанков · построен "
            f"{search_pipeline.indexed_at} · модель `{search_pipeline.index_model}`"
        )

        with st.form("search_form", border=True):
            query = st.text_input(
                "Поисковый запрос",
                placeholder="Например: лифт не работает, куда обращаться?",
            )
            top_k = st.slider(
                "Сколько чанков показать (top-k)", 1, TOP_K_MAX, TOP_K_DEFAULT
            )
            submitted = st.form_submit_button("🔍 Искать", type="primary")

        if submitted:
            if query.strip():
                t0 = time.perf_counter()
                st.session_state["rag_hits"] = search_pipeline.search(
                    query, embedder, k=top_k, instruction=INSTRUCTION
                )
                st.session_state["rag_query"] = query.strip()
                st.session_state["rag_ms"] = (time.perf_counter() - t0) * 1000
            else:
                st.session_state["rag_hits"] = []
                st.warning("Введите непустой запрос.")

        hits = st.session_state.get("rag_hits") or []
        last_query = st.session_state.get("rag_query", "")

        if hits:
            st.success(
                f"Запрос «{last_query}»: найдено {len(hits)} чанков за "
                f"{st.session_state.get('rag_ms', 0):.0f} мс (косинусное сходство)"
            )

            for hit in hits:
                score_pct = max(0.0, min(1.0, (hit.score + 1) / 2))  # для цвета полоски
                color = "#2e7d32" if hit.score >= 0.55 else "#ef6c00" if hit.score >= 0.35 else "#9e9e9e"
                crumb = " > ".join(hit.heading_path) if hit.heading_path else "без заголовка"
                title = f"#{hit.rank} · cos = {hit.score:.3f} · {crumb} · строки {hit.start_line}–{hit.end_line}"
                with st.expander(title, expanded=hit.rank <= 2):
                    st.markdown(
                        f'<span style="color:{color};font-weight:700">'
                        f"косинусное сходство: {hit.score:.3f}</span>",
                        unsafe_allow_html=True,
                    )
                    st.progress(score_pct)
                    st.markdown(hit.text)

            # --- Генерация ответа по найденным чанкам (GigaChat) ------------
            st.divider()
            st.subheader("Ответ на основе найденного (RAG)")

            gigachat_client = get_gigachat()
            if gigachat_client is None:
                st.info(
                    "Генерация не настроена: добавьте `GIGACHAT_CREDENTIALS` в `.env` "
                    "и перезапустите приложение. Поиск работает и без этого."
                )
            elif st.button("🤖 Сгенерировать ответ (GigaChat)", type="primary"):
                generation = GenerationPipeline(
                    score_threshold=config.RELEVANCE_THRESHOLD,
                    max_hits=config.GENERATION_CONTEXT_HITS,
                    temperature=config.GENERATION_TEMPERATURE,
                    max_tokens=config.GENERATION_MAX_TOKENS,
                    model_name=config.GIGACHAT_MODEL,
                )
                try:
                    with st.spinner(f"GigaChat ({config.GIGACHAT_MODEL}) думает…"):
                        result = generation.run(last_query, hits, gigachat_client)
                except LLMError as e:
                    st.error(f"Не удалось получить ответ от GigaChat: {e}")
                else:
                    if result.no_relevant:
                        st.warning(
                            f"Ни один чанк не прошёл порог релевантности "
                            f"({config.RELEVANCE_THRESHOLD:.2f}) — в базе, похоже, нет "
                            f"данных по этому вопросу. Генерация отменена, чтобы "
                            f"избежать выдумок."
                        )
                    else:
                        st.markdown(result.answer)
                        used = " ".join(f"`[{h.rank}]`" for h in result.used_hits)
                        st.caption(
                            f"Опора на фрагменты: {used} · пропущено по порогу "
                            f"релевантности: {result.skipped_low_score} · модель "
                            f"{config.GIGACHAT_MODEL}"
                        )

st.divider()
st.caption(
    "Косинусное сходство реализовано как Inner Product над L2-нормализованными "
    f"BGE-векторами (faiss.IndexFlatIP, точный поиск). Модель: {EMBEDDING_MODEL}."
)
