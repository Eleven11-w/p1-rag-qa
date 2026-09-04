"""D4 向量化与检索：为分块生成向量、建索引，并实现 query -> Top-K 召回。

输出（data/ 下，本地保存）：
  embeddings.npz      向量矩阵 (N, D)，L2 归一化
  index_meta.jsonl    每个向量对应的信息（source/page/chunk_id/chars/text）

用法：
  python scripts/vectorize_index.py                  # 建索引并跑内置自测
  python scripts/vectorize_index.py --q "..." --k 5  # 单条查询

说明：默认对语义切块（chunks_semantic.jsonl）建索引；检索用余弦相似度，
对本规模（几千个向量）直接暴力计算即可，无需专门向量库。
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from sentence_transformers import SentenceTransformer


sys.stdout.reconfigure(encoding="utf-8", errors="replace")


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
CHUNKS = DATA / "chunks_semantic.jsonl"
NPZ = DATA / "embeddings.npz"
META = DATA / "index_meta.jsonl"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "
MIN_CHARS = 60
DIGIT_THRESHOLD = 0.6


def digit_heavy(text: str) -> bool:
    if not text.strip():
        return False
    return sum(c.isdigit() or c.isspace() for c in text) / len(text) > DIGIT_THRESHOLD


def load_chunks() -> tuple[list[dict], int]:
    items = [json.loads(line) for line in CHUNKS.read_text(encoding="utf-8").splitlines()]
    kept = []
    dropped = 0
    for it in items:
        text = it.get("text", "")
        if len(text.strip()) < MIN_CHARS or digit_heavy(text):
            dropped += 1
            continue
        kept.append(it)
    return kept, dropped


def build() -> tuple[SentenceTransformer, np.ndarray, list[dict]]:
    chunks, dropped = load_chunks()
    print(f"chunks loaded={len(chunks)} dropped(短块/纯数字)={dropped}")

    model = SentenceTransformer(MODEL_NAME)
    texts = [c["text"] for c in chunks]
    embs = model.encode(
        texts, batch_size=32, show_progress_bar=False, normalize_embeddings=True
    )
    embs = np.asarray(embs, dtype=np.float32)

    np.savez(NPZ, embeddings=embs)
    with META.open("w", encoding="utf-8") as fh:
        for c in chunks:
            rec = {k: c.get(k) for k in ("source", "title", "arxiv", "page", "chunk_id", "chars")}
            rec["text"] = c["text"]
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"saved embeddings {embs.shape[0]} x {embs.shape[1]} -> {NPZ.name}")
    return model, embs, chunks


def retrieve(model, embs, chunks, query: str, k: int = 5):
    q = model.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
    sims = embs @ q
    order = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in order]


def run_query(model, embs, chunks, query: str, k: int) -> None:
    print(f"\nQ: {query}")
    for i, score in retrieve(model, embs, chunks, query, k):
        c = chunks[i]
        snippet = c["text"][:70].replace("\n", " ")
        print(f"  [{score:.3f}] {c['source'][:36]} p{c['page']} | {snippet}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--q", default=None)
    ap.add_argument("--k", type=int, default=5)
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    if args.q:
        model = SentenceTransformer(MODEL_NAME)
        embs = np.load(NPZ)["embeddings"]
        chunks = [json.loads(line) for line in META.read_text(encoding="utf-8").splitlines()]
        run_query(model, embs, chunks, args.q, args.k)
        return

    model, embs, chunks = build()
    tests = [
        "How is the conditional average treatment effect (CATE) defined?",
        "What is the difference between confounding and selection bias?",
        "How does representation learning handle treatment effect estimation?",
        "What is partial clustering in randomized trials?",
        "How do Gaussian processes handle few-placebo CATE inference?",
    ]
    for q in tests:
        run_query(model, embs, chunks, q, args.k)


if __name__ == "__main__":
    main()
