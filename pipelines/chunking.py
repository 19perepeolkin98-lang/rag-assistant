"""Markdown-чанкер: превращает один большой документ в список чанков.

Стратегия:
1. Документ режется на секции по заголовкам ATX (`#` .. `######`).
   Заголовки внутри код-блоков (``` / ~~~) не считаются заголовками.
2. Секция нарезается на чанки до `chunk_size` символов: сначала по абзацам
   (пустая строка = граница), абзац длиннее лимита режется скользящим окном.
3. Соседние чанки перекрываются на `chunk_overlap` символов, чтобы фраза на
   границе не терялась для поиска.
4. Каждому чанку prepend'ится «хлебная крошка» — путь заголовков секции
   (например, `[ЖКХ > Лифты]`): контекст заголовков заметно улучшает поиск.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

HEADING_RE = re.compile(r"^(#{1,6})\s+(\S.*?)\s*#*\s*$")
FENCE_RE = re.compile(r"^(`{3,}|~{3,})")

MIN_CHUNK_SIZE = 50  # защита от бессмысленных значений из UI


@dataclass
class Chunk:
    text: str                  # итоговый текст (крошка + тело) — он эмбеддится
    heading_path: list[str]    # ["Раздел", "Подраздел"]
    start_line: int            # первая строка секции (1-based)
    end_line: int              # последняя строка секции


@dataclass
class _Section:
    heading_path: list[str]
    start_line: int
    end_line: int
    text: str                  # включая строку самого заголовка


def _close_fence(line: str, fence: str) -> bool:
    """Строка закрывает открытый код-блок, если начинается с того же маркера."""
    return line.strip().startswith(fence)


def split_sections(md_text: str) -> list[_Section]:
    """Режет документ на секции по заголовкам, не заходя внутрь код-блоков."""
    sections: list[_Section] = []
    stack: list[tuple[int, str]] = []   # [(уровень, заголовок)]
    buf: list[str] = []
    buf_start: int | None = None
    fence: str | None = None

    def flush(end_line: int) -> None:
        text = "\n".join(buf).strip("\n").strip()
        if text and buf_start is not None:
            path = [title for _, title in stack]
            sections.append(_Section(path, buf_start, end_line, text))

    for lineno, line in enumerate(md_text.splitlines(), 1):
        if fence is not None:
            buf.append(line)
            if _close_fence(line, fence):
                fence = None
            continue

        m = FENCE_RE.match(line.strip())
        if m:
            buf.append(line)
            fence = m.group(1)
            if buf_start is None:
                buf_start = lineno
            continue

        hm = HEADING_RE.match(line)
        if hm:
            flush(lineno - 1)
            level, title = len(hm.group(1)), hm.group(2).strip()
            while stack and stack[-1][0] >= level:
                stack.pop()
            stack.append((level, title))
            buf, buf_start = [line], lineno
        else:
            buf.append(line)
            if buf_start is None:
                buf_start = lineno

    flush(md_text.count("\n") + 1 if md_text else 1)
    return sections


def _split_paragraphs(text: str) -> list[str]:
    """Абзацы по пустым строкам; код-блок не разрывается изнутри."""
    paragraphs: list[str] = []
    buf: list[str] = []
    fence: str | None = None

    for line in text.splitlines():
        if fence is not None:
            buf.append(line)
            if _close_fence(line, fence):
                fence = None
            continue
        m = FENCE_RE.match(line.strip())
        if m:
            buf.append(line)
            fence = m.group(1)
            continue
        if not line.strip():
            if buf:
                paragraphs.append("\n".join(buf).strip("\n"))
                buf = []
        else:
            buf.append(line)

    if buf:
        paragraphs.append("\n".join(buf).strip("\n"))
    return [p for p in paragraphs if p.strip()]


def _tail(text: str, overlap: int) -> str:
    """Хвост предыдущего чанка для перекрытия; срезаем по границе слова."""
    if overlap <= 0 or not text:
        return ""
    tail = text[-overlap:].strip()
    if not tail:
        return ""
    sp = tail.find(" ")
    if 0 < sp < len(tail) - 1:
        tail = tail[sp + 1:]
    return tail


def _hard_split(paragraph: str, chunk_size: int, overlap: int) -> list[str]:
    """Скользящее окно по слишком длинному абзацу (таблица, код, простыня текста)."""
    step = max(1, chunk_size - overlap)
    pieces: list[str] = []
    covered_until = 0
    i = 0
    while i < len(paragraph):
        if i + chunk_size <= covered_until:   # окно целиком внутри прошлого чанка
            i += step
            continue
        piece = paragraph[i:i + chunk_size].strip()
        if piece:
            pieces.append(piece)
            covered_until = i + chunk_size
        i += step
    return pieces


def _chunk_section_body(body: str, chunk_size: int, overlap: int) -> list[str]:
    if len(body) <= chunk_size:
        return [body.strip()] if body.strip() else []

    pieces: list[str] = []
    buf = ""
    for para in _split_paragraphs(body):
        if len(para) > chunk_size:
            if buf.strip():
                pieces.append(buf)
            pieces.extend(_hard_split(para, chunk_size, overlap))
            buf = ""
            continue
        candidate = para if not buf else buf + "\n\n" + para
        if len(candidate) <= chunk_size:
            buf = candidate
            continue
        pieces.append(buf)
        seed = _tail(buf, overlap)
        new_buf = seed + "\n\n" + para if seed else para
        if len(new_buf) <= chunk_size:
            buf = new_buf
        else:  # абзац почти в размер лимита — перекрытие уже не влезает
            pieces.append(para)
            buf = ""
    if buf.strip():
        pieces.append(buf)
    return [p for p in pieces if p.strip()]


def chunk_markdown(
    md_text: str,
    chunk_size: int,
    chunk_overlap: int,
) -> list[Chunk]:
    """Главная функция: Markdown → список чанков для эмбеддинга."""
    if chunk_size < MIN_CHUNK_SIZE:
        raise ValueError(f"chunk_size должен быть не меньше {MIN_CHUNK_SIZE}")
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap должен быть строго меньше chunk_size")
    if not md_text.strip():
        return []

    chunks: list[Chunk] = []
    for section in split_sections(md_text):
        crumb = " > ".join(section.heading_path)
        for piece in _chunk_section_body(section.text, chunk_size, chunk_overlap):
            text = f"[{crumb}]\n\n{piece}" if crumb else piece
            chunks.append(
                Chunk(
                    text=text,
                    heading_path=list(section.heading_path),
                    start_line=section.start_line,
                    end_line=section.end_line,
                )
            )
    return chunks
