"""D2 文档解析：把 data/*.pdf 逐个转成带来源标注的 JSONL。

输出：data/corpus.jsonl，每行一条记录：
    {source, title, arxiv, page, chars, text}

用法：在项目根目录运行
    python scripts/parse_docs.py

说明：text 为按页提取的文本；后续 D3 切块、D6 评测都会以
source + page 作为引用依据（溯源）。
"""

import json
import re
import sys
from pathlib import Path

import pymupdf  # PyMuPDF


sys.stdout.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = DATA / "corpus.jsonl"

ARXIV_RE = re.compile(
    r"arXiv\s*:\s*([0-9]{4}\.[0-9]{4,5}(?:v[0-9]+)?)", re.IGNORECASE
)


def normalize(text: str) -> str:
    """统一换行、压缩连续空行，保留段落边界。"""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def main() -> None:
    pdfs = sorted(DATA.glob("*.pdf"))
    if not pdfs:
        sys.exit("data 下没有 PDF。")

    total_pages = 0
    total_chars = 0
    low_chars = []

    with OUT.open("w", encoding="utf-8") as fh:
        for pdf in pdfs:
            with pymupdf.open(pdf) as doc:
                meta = doc.metadata or {}
                title = (meta.get("title") or "").strip()

                head = "\n".join(doc[i].get_text() for i in range(min(2, doc.page_count)))
                m = ARXIV_RE.search(head)
                arxiv = m.group(1) if m else ""

                for page in doc:
                    text = normalize(page.get_text("text"))
                    chars = len(text)
                    total_pages += 1
                    total_chars += chars
                    if chars < 100:
                        low_chars.append(
                            {"source": pdf.name, "page": page.number + 1, "chars": chars}
                        )
                    rec = {
                        "source": pdf.name,
                        "title": title or (text.splitlines()[0] if text else ""),
                        "arxiv": arxiv,
                        "page": page.number + 1,
                        "chars": chars,
                        "text": text,
                    }
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"PDFs={len(pdfs)} pages={total_pages} chars={total_chars} lowCharPages={len(low_chars)}")
    print("output:", OUT)
    if low_chars:
        print("极短页(可能为扫描/空白，需检查):")
        for item in low_chars[:20]:
            print(f"  {item['source']} page={item['page']} chars={item['chars']}")


if __name__ == "__main__":
    main()
