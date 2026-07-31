"""Direct Google Gemini API model wrapper."""

from typing import Dict, Any

from .base_model import BaseModel, get_shared_client


class GeminiModel(BaseModel):
    """Calls the Google Gemini generateContent API directly."""

    API_URL_TEMPLATE = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(self, model_id: str, name: str = "Gemini"):
        super().__init__(name, model_id, "GOOGLE_API_KEY")

    async def classify_content(
        self, data: Dict[str, str], prompt: str
    ) -> Dict[str, Any]:
        try:
            url = self.API_URL_TEMPLATE.format(model=self.model_id)
            payload = {
                "contents": [
                    {
                        "parts": [
                            {"text": f"{prompt}\n\n{data['content']}"}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0,
                    "maxOutputTokens": 8192,
                },
            }
            client = await get_shared_client()
            response = await client.post(
                url, params={"key": self.api_key}, json=payload,
            )
            response.raise_for_status()
            text = response.json()["candidates"][0]["content"]["parts"][0]["text"]
            return self._parse_response(text)
        except Exception as e:
            return {"error": f"{type(e).__name__}: {e}"}
