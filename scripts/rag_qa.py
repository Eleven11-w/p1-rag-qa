"""D5 基线问答：检索 + 依据资料生成带引用的回答。

流程：问题 -> 向量检索 Top-K -> 拼成带编号的上下文 -> 调用对话模型
      -> 模型只依据资料回答并标注 [n] 引用。

用法：python scripts/rag_qa.py
"""

import json
import os
import sys
from pathlib import Path

import numpy as np
from dotenv import load_dotenv
from openai import OpenAI
from sentence_transformers import SentenceTransformer


sys.stdout.reconfigure(encoding="utf-8", errors="replace")


load_dotenv()

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
NPZ = DATA / "embeddings.npz"
META = DATA / "index_meta.jsonl"

MODEL_NAME = "BAAI/bge-small-en-v1.5"
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

client = OpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"),
)
chat_model = os.environ.get("LLM_MODEL", "deepseek-chat")


def load_index():
    embs = np.load(NPZ)["embeddings"]
    meta = [json.loads(line) for line in META.read_text(encoding="utf-8").splitlines()]
    return embs, meta


def retrieve(encoder, embs, query: str, k: int = 5):
    qv = encoder.encode([QUERY_PREFIX + query], normalize_embeddings=True)[0]
    sims = embs @ qv
    order = np.argsort(-sims)[:k]
    return [(int(i), float(sims[i])) for i in order]


def build_context(meta, hits) -> str:
    parts = []
    for n, (i, score) in enumerate(hits, 1):
        c = meta[i]
        parts.append(f"[{n}] 来源：{c['source']} 第{c['page']}页\n{c['text']}")
    return "\n\n".join(parts)


def ask(question: str, context: str):
    system = (
        "你是严谨的中文学术问答助手。只依据‘资料’回答，不得使用资料之外的知识；"
        "若资料不足以回答，请明确说‘资料中没有相关信息’。"
        "先给出结论，再在关键句后用 [n] 标注引用，n 对应资料编号。"
    )
    resp = client.chat.completions.create(
        model=chat_model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": f"资料：\n{context}\n\n问题：{question}"},
        ],
        temperature=0.2,
    )
    return resp.choices[0].message.content, resp.usage.total_tokens


def main() -> None:
    embs, meta = load_index()
    encoder = SentenceTransformer(MODEL_NAME)
    print(f"index: {embs.shape[0]} chunks")

    questions = [
        "条件平均处理效应（CATE）是怎么定义的？",
        "混杂偏倚（confounding）和选择偏倚（selection bias）的区别是什么？",
        "表示学习（representation learning）如何用于治疗效果估计？",
        "随机试验中的部分聚类（partial clustering）是什么意思？",
        "少安慰剂情形下，高斯过程如何做 CATE 的校准推断？",
        "比率型治疗效果（ratio-based treatment effects）的双稳健元学习器是什么？",
        "DFW 这种加权方案如何做协变量平衡和效应估计？",
        "通过辅助变量实现反事实公平（counterfactual fairness）的思路是什么？",
        "网络数据上的因果效应估计，什么时候选择预测、什么时候选择拒绝？",
        "如何制作川菜宫保鸡丁？",
    ]

    for qi, q in enumerate(questions, 1):
        hits = retrieve(encoder, embs, q, k=5)
        context = build_context(meta, hits)
        answer, tokens = ask(q, context)
        print(f"\n===== Q{qi}: {q} =====")
        for n, (i, score) in enumerate(hits, 1):
            c = meta[i]
            print(f"  [{n}] ({score:.3f}) {c['source'][:40]} 第{c['page']}页")
        print(f"--- 回答 (tokens={tokens}) ---")
        print(answer)


if __name__ == "__main__":
    main()
