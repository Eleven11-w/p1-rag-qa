"""D1 最小示例：跑通 OpenAI 兼容 API，并让模型返回 JSON。

用法：
1. 复制 .env.example 为 .env，填入 LLM_API_KEY
2. pip install -r requirements.txt
3. python src/day1_structured_demo.py

若所用模型不支持 response_format 参数，删掉该参数，
并在 system 提示词中强调“只输出 JSON”。
"""

import os

from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()

client = OpenAI(
    api_key=os.environ["LLM_API_KEY"],
    base_url=os.environ.get("LLM_BASE_URL", "https://api.deepseek.com"),
)
model = os.environ.get("LLM_MODEL", "deepseek-chat")

resp = client.chat.completions.create(
    model=model,
    messages=[
        {
            "role": "system",
            "content": "你是结构化输出助手，只输出合法的 JSON，不要输出其他内容。",
        },
        {
            "role": "user",
            "content": (
                "请用中文介绍 RAG（检索增强生成）。"
                '返回 JSON，字段为：{"topic": 一句话主题, '
                '"one_line": 用一句话解释 RAG, '
                '"two_reasons": 使用 RAG 的两个理由}'
            ),
        },
    ],
    response_format={"type": "json_object"},
    temperature=0.2,
)

content = resp.choices[0].message.content
print(content)
print("---- usage ----")
print(f"tokens={resp.usage.total_tokens}")
