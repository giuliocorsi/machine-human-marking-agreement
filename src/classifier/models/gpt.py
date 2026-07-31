"""Direct OpenAI API model wrapper."""

from typing import Dict, Any

from .base_model import BaseModel, get_shared_client


class GptModel(BaseModel):
    """Calls the OpenAI Chat Completions API directly."""

    API_URL = "https://api.openai.com/v1/chat/completions"

    def __init__(self, model_id: str, name: str = "GPT"):
        super().__init__(name, model_id, "OPENAI_API_KEY")

    async def classify_content(
        self, data: Dict[str, str], prompt: str
    ) -> Dict[str, Any]:
        try:
            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            }
            payload = {
                "model": self.model_id,
                "messages": [
                    {"role": "system", "content": "You are an expert marker of undergraduate psychology essays in the UK."},
                    {"role": "user", "content": f"{prompt}\n\n{data['content']}"},
                ],
                "temperature": 0,
                "max_completion_tokens": 4096,
            }
            client = await get_shared_client()
            response = await client.post(
                self.API_URL, headers=headers, json=payload,
            )
            response.raise_for_status()
            text = response.json()["choices"][0]["message"]["content"]
            return self._parse_response(text)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
