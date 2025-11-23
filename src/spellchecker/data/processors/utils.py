import os
import typing as tp

import openai

YANDEX_CLOUD_FOLDER = os.getenv("YANDEX_CLOUD_FOLDER")
YANDEX_CLOUD_API_KEY = os.getenv("YANDEX_CLOUD_API_KEY")

assert YANDEX_CLOUD_FOLDER, "YANDEX_CLOUD_FOLDER env variable is not set"
assert YANDEX_CLOUD_API_KEY, "YANDEX_CLOUD_API_KEY env variable is not set"

_client = openai.OpenAI(
    api_key=YANDEX_CLOUD_API_KEY, base_url="https://llm.api.cloud.yandex.net/v1"
)


def query_llm(
    prompt: str,
    system_prompt: str = "",
    model: str = "qwen3-235b-a22b-fp8/latest",
    max_tokens: int = 500,
    temperature: float = 0.3,
    stream: bool = False,
) -> tp.Generator[str, None, None]:
    """Query Yandex Cloud LLM and yield response chunks."""
    response = _client.chat.completions.create(
        model=f"gpt://{YANDEX_CLOUD_FOLDER}/{model}",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        max_tokens=max_tokens,
        temperature=temperature,
        stream=stream,
    )

    if stream:
        for chunk in response:
            if chunk.choices[0].delta.content is not None:
                yield chunk.choices[0].delta.content
    else:
        yield response.choices[0].message.content
