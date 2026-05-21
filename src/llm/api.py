import json
import re
from typing import List, Optional

from src.utils.logger import get_logger

logger = get_logger("llm_api")


class LLMClient:
    def __init__(self, provider: str = "openai", model: str = "gpt-4o-mini", api_key: Optional[str] = None,
                 base_url: Optional[str] = None):
        self.provider = provider
        self.model = model

        if provider == "openai":
            from openai import OpenAI
            kwargs = {}
            if api_key:
                kwargs["api_key"] = api_key
            if base_url:
                kwargs["base_url"] = base_url
            self.client = OpenAI(**kwargs)
        else:
            raise ValueError(f"Unsupported provider: {provider}")

    def chat(self, system: str, user: str, temperature: float = 0.0, max_tokens: int = 500) -> str:
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return response.choices[0].message.content.strip()
        except Exception as e:
            logger.error(f"LLM API error: {e}")
            return ""

    def rank(self, system: str, user_prompt: str, num_candidates: int) -> List[int]:
        response = self.chat(system, user_prompt, temperature=0.0)
        return self._parse_ranking(response, num_candidates)

    def _parse_ranking(self, text: str, num_candidates: int) -> List[int]:
        numbers = re.findall(r"\d+", text)
        seen = set()
        result = []
        for num_str in numbers:
            num = int(num_str)
            if 1 <= num <= num_candidates and num not in seen:
                seen.add(num)
                result.append(num)
        for i in range(1, num_candidates + 1):
            if i not in seen:
                result.append(i)
        return result
