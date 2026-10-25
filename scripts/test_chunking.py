# -*- coding: utf-8 -*-
"""Юнит-проверки Markdown-чанкера (без загрузки модели).

Запуск: python scripts/test_chunking.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipelines.chunking import (  # noqa: E402
    Chunk,
    _chunk_section_body,
    chunk_markdown,
    split_sections,
)

FAILURES: list[str] = []


def check(name: str, cond: bool, detail: str = "") -> None:
    status = "OK " if cond else "FAIL"
    print(f"[{status}] {name}" + (f" — {detail}" if detail and not cond else ""))
    if not cond:
        FAILURES.append(name)


# --- 1. Секции по заголовкам -------------------------------------------------
md = (
    "# Заголовок 1\nтекст до подвала\n\n## Подраздел A\nтекст A\n\n"
    "## Подраздел B\nтекст B\n"
)
sections = split_sections(md)
check("секции: 3 штуки", len(sections) == 3, f"получено {len(sections)}")
check(
    "секции: пути заголовков",
    [s.heading_path for s in sections]
    == [["Заголовок 1"], ["Заголовок 1", "Подраздел A"], ["Заголовок 1", "Подраздел B"]],
)
check(
    "секции: текст включает заголовок",
    sections[1].text.startswith("## Подраздел A"),
)

# --- 2. Заголовки внутри код-блоков игнорируются ------------------------------
md_fence = (
    "# Реал\n\n```python\n# это не заголовок\ndef f():\n    pass\n```\n\n"
    "после кода\n"
)
sections = split_sections(md_fence)
check("код-блок: одна секция", len(sections) == 1, f"получено {len(sections)}")
check("код-блок: содержимое сохранено", "def f():" in sections[0].text)

# --- 3. Абзацы внутри код-блока не разрываются пустыми строками ---------------
md_code_paras = "код:\n```sql\nSELECT 1;\n\nSELECT 2;\n```\nконец"
paras = _chunk_section_body(md_code_paras, chunk_size=10_000, overlap=0)
check("код-блок не рвётся по абзацам", len(paras) == 1 and "SELECT 2;" in paras[0])

# --- 4. Размер чанков и перекрытие --------------------------------------------
long_section = "\n\n".join(
    f"Абзац номер {i} с каким-то содержательным текстом." for i in range(60)
)
pieces = _chunk_section_body("# T\n" + long_section, chunk_size=500, overlap=100)
check(
    "размер: все чанки <= лимита",
    all(len(p) <= 500 for p in pieces),
    f"max = {max(map(len, pieces))}",
)
check("перекрытие: чанков больше одного", len(pieces) > 1)
has_overlap = any(
    pieces[i][-60:] in pieces[i + 1] or pieces[i + 1][:60] in pieces[i]
    for i in range(len(pieces) - 1)
)
check("перекрытие: соседние чанки пересекаются", has_overlap)
check(
    "полнота: контент не теряется",
    all(w in "".join(pieces) for w in ["Абзац номер 0", "Абзац номер 59"]),
)

# --- 5. Слишком длинный абзац режется скользящим окном -------------------------
huge_para = "слово " * 3000
pieces = _chunk_section_body(huge_para, chunk_size=400, overlap=80)
check("hard split: несколько кусков", len(pieces) > 5, f"получено {len(pieces)}")
check("hard split: размер в лимите", all(len(p) <= 400 for p in pieces))
check("hard split: конец абзаца дошёл", "слово слово" in pieces[-1])

# --- 6. Полный пайплайн: хлебные крошки ----------------------------------------
chunks = chunk_markdown(md, chunk_size=1000, chunk_overlap=200)
check(
    "крошки: текст начинается с пути",
    chunks[1].text.startswith("[Заголовок 1 > Подраздел A]"),
)
check(
    "крошки: метаданные на месте",
    isinstance(chunks[0], Chunk) and chunks[0].start_line == 1,
)

# --- 7. Защита от неверных параметров -------------------------------------------
try:
    chunk_markdown("текст", chunk_size=1000, chunk_overlap=1000)
    check("валидация: overlap >= size падает", False)
except ValueError:
    check("валидация: overlap >= size падает", True)

print()
if FAILURES:
    print("ПРОВАЛЕНО:", len(FAILURES), "->", FAILURES)
    sys.exit(1)
print("Все проверки чанкера пройдены")
