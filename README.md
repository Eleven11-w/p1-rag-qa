# p1-rag-qa · 领域知识库问答（RAG）

> 状态：D1 已完成（2026-09-03）
> 默认领域：因果推断 / 科研方法论（使用可公开资料）
> 一句话：给定一批领域文档，用户提问后系统检索相关原文，并给出带来源引用的回答。

## 目录规划

```text
p1-rag-qa/
├─ README.md
├─ .gitignore
├─ .env.example      # 复制为 .env 后填密钥
├─ requirements.txt  # 依赖
├─ data/             # 语料（本地存放，不提交 GitHub）
├─ scripts/          # 解析、切块、索引、检索、评测脚本（按 D2–D15 逐步填充）
├─ app/              # 问答服务与界面
└─ eval/             # 评测集与评测报告
```

## 技术栈

- Python + OpenAI 兼容 SDK；API 默认 DeepSeek（可换 Qwen/GLM）
- Embedding：bge-m3；Reranker：bge-reranker
- 向量库：FAISS 或 Chroma；混合检索：BM25 + Dense
- 界面：Gradio / Streamlit；部署：Docker（D13 前补齐）

## 评测门禁目标

自建 80 题评测集（40 单点定位 + 20 跨章节综合 + 20 表格/数值）：

- 依据块 Top5 检索命中率 ≥ 85%
- 端到端“完全正确 + 部分正确”≥ 80%，完全正确 ≥ 60%
- 引用来源抽检可溯源 ≥ 90%

## D1 启动清单

- [x] 确定语料：24 篇因果推断论文入库（清单见 data/语料清单.md）
- [x] GitHub 网页创建同名空仓库 `p1-rag-qa`
- [x] 复制 `.env.example` 为 `.env` 并填入 API 密钥
- [x] 安装依赖：`pip install -r requirements.txt`（venv 已建好）
- [x] 运行结构化输出示例：输出 JSON，tokens=274
- [x] 完成首次 commit 并 push

## 本地运行示例（D1）

```powershell
Copy-Item .env.example .env
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python src/day1_structured_demo.py
```

> 密钥只写进 `.env`，该文件已被 `.gitignore` 排除。
