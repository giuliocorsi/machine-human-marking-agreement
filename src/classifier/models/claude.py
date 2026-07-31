"""Direct Anthropic Claude API model wrapper."""

from typing import Dict, Any

from .base_model import BaseModel, get_shared_client


class ClaudeModel(BaseModel):
    """Calls the Anthropic Messages API directly."""

    API_URL = "https://api.anthropic.com/v1/messages"

    def __init__(self, model_id: str, name: str = "Claude"):
        super().__init__(name, model_id, "ANTHROPIC_API_KEY")

    async def classify_content(
        self, data: Dict[str, str], prompt: str
    ) -> Dict[str, Any]:
        try:
            headers = {
                "x-api-key": self.api_key,
                "anthropic-version": "2023-06-01",
                "content-type": "application/json",
            }
            payload = {
                "model": self.model_id,
                "max_tokens": 4096,
                "temperature": 0,
                "system": "You are an expert marker of undergraduate psychology essays in the UK.",
                "messages": [
                    {"role": "user", "content": f"{prompt}\n\n{data['content']}"}
                ],
            }
            client = await get_shared_client()
            response = await client.post(
                self.API_URL, headers=headers, json=payload,
            )
            response.raise_for_status()
            text = response.json()["content"][0]["text"]
            return self._parse_response(text)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
