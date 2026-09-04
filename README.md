# p1-rag-qa · 领域知识库问答（RAG）

> 状态：D1–D5 已完成（截至 2026-09-04）
> 默认领域：因果推断 / 科研方法论（使用可公开资料）
> 一句话：给定一批领域文档，用户提问后系统检索相关原文，并给出带来源引用的回答。

## 项目简介

这是一个从零到一实现的“论文知识库问答”系统：把论文 PDF 解析成带来源的结构化文本，切成语义片段，转成向量建索引，最后让大模型**只依据检索到的原文**回答并标注出处。它演示 RAG 的完整链路，以及“如何用评测数据而不是感觉来优化”。

## 整体流程

```text
论文 PDF ── D2 解析 ──> corpus.jsonl ── D3 切块 ──> chunks_*.jsonl
                                                  │
                                                  └─ D4 向量化 ──> embeddings.npz + index_meta.jsonl
                                                            │
D5 问答 ──> 问题 ──> 检索 Top-5 ──> 拼上下文 ──> 模型带引用回答
```

## 技术栈（当前实际使用）

- Python；OpenAI 兼容 SDK（对话默认 DeepSeek，可换 Qwen/GLM）
- 文档解析：PyMuPDF（`pymupdf`）
- 向量化：`sentence-transformers` + `bge-small-en-v1.5`（本地模型，英语为主）
- 检索：余弦相似度 + numpy（当前几千个向量规模，用暴力计算；规模增大再上 FAISS/ANN）
- 后续升级方向：bge-m3（双语）、BM25 混合检索、reranker、Docker 部署

## 目录结构

```text
p1-rag-qa/
├─ README.md
├─ .env.example        # 复制为 .env 后填密钥
├─ requirements.txt
├─ data/               # 语料与中间产物（本地，gitignore）
│  ├─ 语料清单.md
│  ├─ corpus.jsonl     # D2 解析结果
│  ├─ chunks_*.jsonl   # D3 切块结果
│  └─ embeddings.npz   # D4 向量
├─ src/
│  └─ day1_structured_demo.py
├─ scripts/
│  ├─ parse_docs.py       # D2
│  ├─ chunk_docs.py       # D3
│  ├─ vectorize_index.py  # D4
│  └─ rag_qa.py           # D5
└─ eval/               # 评测集与报告（D6 起）
```

## 快速开始（D1–D5 操作说明）

### 0. 前置环境

```powershell
cd "E:\03_Development\AI application and development\p1-rag-qa"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
Copy-Item .env.example .env
notepad .env          # 填入 LLM_API_KEY
```

首次运行向量化脚本会自动下载 embedding 模型（约 130MB）。国内网络可在运行前设置镜像：

```powershell
$env:HF_ENDPOINT = "https://hf-mirror.com"
```

### 1. D1 · 语料与 API 验证

把领域文档放进 `data\`（默认 24 篇因果推断论文，`data/语料清单.md` 记录每篇的来源与量级）。验证对话 API 能返回结构化 JSON：

```powershell
python src\day1_structured_demo.py
```

预期输出：一段 JSON + `tokens=274` 左右的用量。

### 2. D2 · 文档解析（PDF → 结构化原文）

```powershell
python scripts\parse_docs.py
```

输出 `data/corpus.jsonl`，每行 = 某篇论文的某页，字段含 `source`（论文文件名）、`page`、`title`、`arxiv`、`chars`、`text`。脚本会打印总页数、字符数，并提示“近空白页”（通常是插图/分节页，需人工确认）。

### 3. D3 · 切块（chunking）

```powershell
python scripts\chunk_docs.py
```

生成两套分块：

- `data/chunks_fixed.jsonl`：固定长度 800 字符 + 100 重叠
- `data/chunks_semantic.jsonl`：按段落合并（目标 800，上限 1600）

并在控制台输出每种策略的块数、平均/最大/最短长度、短块与“数字密集”块数量，便于对比。

### 4. D4 · 向量化与检索

```powershell
# 建索引（首次较慢，结果缓存）
python scripts\vectorize_index.py

# 单条查询
python scripts\vectorize_index.py --q "how is CATE defined" --k 5
```

索引结果：`data/embeddings.npz`（向量矩阵）与 `data/index_meta.jsonl`（每个向量对应的来源/页码）。脚本内置 5 个英文测试问题，验证 Top-5 是否召回对应论文。

### 5. D5 · 检索增强问答

```powershell
python scripts\rag_qa.py
```

流程：问题 → 检索 Top-5 → 拼成带编号资料 → 调用大模型生成回答。Prompt 有三条约束：只依据资料回答、资料不足就明说、答案用 `[n]` 标注出处。脚本输出每个问题的候选来源、模型回答与 token 用量。

## 数据与版权说明

- `data/` 内的论文 PDF 与解析出的全文、分块、向量均为本地文件，`gitignore` 排除，不进入 GitHub；
- 原因：论文版权不随 arXiv 开放而自动放开，公开仓库不放全文；
- 后续复现方案：写一个按 arXiv ID 自动下载的脚本，或放少量自写摘要作为公开样例。

## 评测门禁（规划，D6 起实现）

自建 80 题评测集（40 单点定位 + 20 跨章节综合 + 20 表格/数值）：

- 依据块 Top5 检索命中率 ≥ 85%
- 端到端“完全正确 + 部分正确”≥ 80%，完全正确 ≥ 60%
- 引用来源抽检可溯源 ≥ 90%

## 当前进度与已知问题

- D1–D5 已完成：解析、切块、向量化、检索、带引用问答均跑通；
- 已知问题（来自 D5 实测）：当前 embedding 是英文模型 `bge-small-en-v1.5`，用中文问题检索英文语料时，部分本应命中的论文无法召回。接下来会通过“提问前翻译”或“换双语 bge-m3”做对照实验，这正是 RAG 里“检索层决定上限”的一个真实案例。
- 下一步：D6 建评测集并用 Recall@5、答案正确率、引用可信度量化效果。

## 技术说明

- 密钥只放 `.env`，已被 `.gitignore` 排除；
- 到 `检索` 这一步先用 numpy 暴力算余弦相似度，是为了先讲清原理；规模化后用 FAISS 等 ANN 索引。
