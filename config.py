"""Центральная конфигурация RAG-приложения.

Любой параметр можно переопределить через файл .env (см. .env.example).
"""
import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:  # python-dotenv не обязателен, если vars заданы в окружении
    pass

BASE_DIR = Path(__file__).resolve().parent

# --- Эмбеддинги ------------------------------------------------------------
# deepvk/USER-bge-m3 — дистилляция BAAI/bge-m3 специально для русского языка
# (VK, 359M параметров, ~1.4 ГБ, dim 768). Англоязычная альтернатива полегче:
# BAAI/bge-small-en-v1.5 (33M, ~130 МБ, dim 384).
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "deepvk/USER-bge-m3")


def query_instruction(model_name: str) -> str:
    """Префикс для поискового запроса (рекомендация авторов BGE).

    Нужен только англоязычным моделям bge-*-en; документам никогда не добавляется.
    """
    name = model_name.lower()
    if "bge" in name and "-en" in name:
        return "Represent this sentence for searching relevant passages: "
    return ""


def is_english_only_model(model_name: str) -> bool:
    name = model_name.lower()
    return "bge" in name and "-en" in name


# --- Чанкинг ---------------------------------------------------------------
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))        # лимит чанка, символов
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))   # перекрытие соседних чанков

# --- Хранилище индекса -----------------------------------------------------
STORAGE_DIR = Path(os.getenv("STORAGE_DIR", str(BASE_DIR / "storage")))
INDEX_FILE = "index.faiss"   # FAISS-индекс (IndexFlatIP по норм. векторам)
CHUNKS_FILE = "chunks.json"  # тексты чанков + метаданные (порядок = id в индексе)
META_FILE = "meta.json"      # модель, размерность, параметры чанкинга, время
CORPUS_FILE = "corpus.md"    # копия последнего проиндексированного корпуса
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "32"))

# --- Поиск -----------------------------------------------------------------
TOP_K_DEFAULT = 5
TOP_K_MAX = 20

# --- Генерация ответа (GigaChat API) ----------------------------------------
GIGACHAT_CREDENTIALS = os.getenv("GIGACHAT_CREDENTIALS")
GIGACHAT_SCOPE = os.getenv("GIGACHAT_SCOPE", "GIGACHAT_API_PERS")
GIGACHAT_MODEL = os.getenv("GIGACHAT_MODEL", "GigaChat-2")
GIGACHAT_TIMEOUT = float(os.getenv("GIGACHAT_TIMEOUT", "120"))
# у GigaChat самоподписанные сертификаты в личном контуре — по умолчанию выключаем проверку
GIGACHAT_VERIFY_SSL = os.getenv("GIGACHAT_VERIFY_SSL", "false").strip().lower() in ("1", "true", "yes")

GENERATION_TEMPERATURE = float(os.getenv("GENERATION_TEMPERATURE", "0.2"))
GENERATION_MAX_TOKENS = int(os.getenv("GENERATION_MAX_TOKENS", "1024"))
# сколько лучших чанков передаём в промпт
GENERATION_CONTEXT_HITS = int(os.getenv("GENERATION_CONTEXT_HITS", "5"))
# косинус ниже порога = чанк нерелевантен; LLM не вызываем, отвечаем «нет данных»
RELEVANCE_THRESHOLD = float(os.getenv("RELEVANCE_THRESHOLD", "0.50"))

# GigaChat не должен ходить через системный прокси/VPN: зарубежный выход
# режется geo-блокировкой Сбера (SSL EOF). Обходим прокси для его доменов.
os.environ.setdefault(
    "NO_PROXY",
    "sberbank.ru,.sberbank.ru,gigachat.devices.sberbank.ru,ngw.devices.sberbank.ru,api.giga.chat,.giga.chat",
)
