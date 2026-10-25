"""Пайплайны RAG: индексация, поиск, генерация, чат-ассистент."""
from pipelines.chat import (
    MODE_CHAT,
    MODE_RAG,
    MODE_REASONING,
    ChatPipeline,
    ChatReply,
)
from pipelines.chunking import Chunk, chunk_markdown
from pipelines.generation import GenerationPipeline, GenerationResult, LLMError
from pipelines.indexing import IndexingPipeline, IndexingStats, model_name_of
from pipelines.router import RouteDecision, route_query
from pipelines.search import (
    IndexNotFoundError,
    ModelMismatchError,
    SearchHit,
    SearchPipeline,
)

__all__ = [
    "MODE_CHAT",
    "MODE_RAG",
    "MODE_REASONING",
    "ChatPipeline",
    "ChatReply",
    "Chunk",
    "chunk_markdown",
    "GenerationPipeline",
    "GenerationResult",
    "LLMError",
    "IndexingPipeline",
    "IndexingStats",
    "model_name_of",
    "RouteDecision",
    "route_query",
    "IndexNotFoundError",
    "ModelMismatchError",
    "SearchHit",
    "SearchPipeline",
]
