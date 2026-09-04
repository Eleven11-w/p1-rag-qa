"""D3 切块：对 data/corpus.jsonl 做两种切块策略并统计对比。

输出（均在 data/ 下，本地保存）：
  chunks_fixed.jsonl    固定长度 + 重叠
  chunks_semantic.jsonl 按段落语义合并

每条 chunk 记录字段：{source, title, arxiv, page, chunk_id, chars, text}
chunk 都落在单一页内，便于后续按 source+page 溯源。

用法：python scripts/chunk_docs.py
"""

import json
import re
import sys
from pathlib import Path


sys.stdout.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CORPUS = DATA / "corpus.jsonl"

FIXED_SIZE = 800
FIXED_OVERLAP = 100
PARA_TARGET = 800
PARA_MAX = 1600


def load_corpus() -> list[dict]:
    return [json.loads(line) for line in CORPUS.read_text(encoding="utf-8").splitlines()]


def chunk_fixed(text: str, size: int = FIXED_SIZE, overlap: int = FIXED_OVERLAP) -> list[str]:
    """按字符窗口切分，回退到词边界，避免从单词中间截断。"""
    out = []
    start = 0
    n = len(text)
    while start < n:
        end = min(start + size, n)
        if end < n:
            sp = text.rfind(" ", start, end)
            if sp > start + size // 2:
                end = sp
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        if end >= n:
            break
        start = end - overlap
    return out


def chunk_semantic(
    text: str, target: int = PARA_TARGET, max_size: int = PARA_MAX
) -> list[str]:
    """按段落切分，贪心合并到接近 target；超大段落再按固定窗口切。"""
    paras = [p.strip() for p in re.split(r"\n\s*\n", text) if p.strip()]
    chunks = []
    cur = ""
    for para in paras:
        if len(para) > max_size:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.extend(chunk_fixed(para, size=max_size, overlap=50))
            continue
        if cur and len(cur) + 1 + len(para) > target:
            chunks.append(cur)
            cur = para
        else:
            cur = (cur + "\n\n" + para) if cur else para
    if cur:
        chunks.append(cur)
    return chunks


def summarize(name: str, chunks: list[str]) -> None:
    lens = [len(c) for c in chunks]
    short = sum(1 for c in chunks if len(c) < 60)
    digit_heavy = sum(
        1
        for c in chunks
        if c.strip() and sum(ch.isdigit() or ch.isspace() for ch in c) / len(c) > 0.6
    )
    avg = sum(lens) / len(lens) if lens else 0
    print(f"[{name}] chunks={len(chunks)} avg={avg:.0f} max={max(lens) if lens else 0} "
          f"min={min(lens) if lens else 0} short(<60)={short} digitHeavy={digit_heavy}")


def main() -> None:
    records = load_corpus()
    print(f"corpus records={len(records)}")

    fixed_out = DATA / "chunks_fixed.jsonl"
    sem_out = DATA / "chunks_semantic.jsonl"

    fixed_chunks: list[str] = []
    sem_chunks: list[str] = []
    skipped = []

    def write_chunks(path: Path, items: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as fh:
            for item in items:
                fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    fixed_items = []
    sem_items = []
    cid = 0
    for rec in records:
        text = rec.get("text", "")
        if not text.strip():
            skipped.append(f"{rec['source']} p{rec['page']}")
            continue
        base = {k: rec.get(k) for k in ("source", "title", "arxiv", "page")}

        for piece in chunk_fixed(text):
            cid += 1
            fixed_items.append({**base, "chunk_id": cid, "chars": len(piece), "text": piece})
        for piece in chunk_semantic(text):
            cid += 1
            sem_items.append({**base, "chunk_id": cid, "chars": len(piece), "text": piece})

    write_chunks(fixed_out, fixed_items)
    write_chunks(sem_out, sem_items)

    print("skipped empty pages:", len(skipped))
    for s in skipped[:5]:
        print("  ", s)
    summarize("fixed", [i["text"] for i in fixed_items])
    summarize("semantic", [i["text"] for i in sem_items])
    print("output:", fixed_out, sem_out)

    print("\n--- fixed sample ---")
    for i in fixed_items[3:5]:
        print(i["source"][:40], f"p{i['page']}", i["chars"], "|", i["text"][:90].replace("\n", " "))
    print("\n--- semantic sample ---")
    for i in sem_items[3:5]:
        print(i["source"][:40], f"p{i['page']}", i["chars"], "|", i["text"][:90].replace("\n", " "))


if __name__ == "__main__":
    main()
