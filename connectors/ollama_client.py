import requests
import logging
import json

logger = logging.getLogger(__name__)

class OllamaClient:
    """Connects to the local Ollama LLM running on the AI PC"""
    def __init__(self, base_url="http://localhost:11434"):
        self.base_url = base_url

    def generate_score(self, prompt: str, model: str = "gemma4:e4b") -> float:
        """Ask LLM to generate a score based on a prompt and parse it."""
        logger.info(f"Querying local LLM ({model}) at {self.base_url}...")
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False
                }
            )
            response.raise_for_status()
            data = response.json()
            
            # This is a very rudimentary parse, you'll likely use Instructor/Pydantic
            raw_text = data.get("response", "")
            logger.debug(f"LLM Raw Response: {raw_text}")
            
            # Simple fallback parsing logic:
            if "score" in raw_text.lower() and "{" in raw_text:
                start = raw_text.find("{")
                end = raw_text.rfind("}") + 1
                json_str = raw_text[start:end]
                result = json.loads(json_str)
                return float(result.get("score", 0.5))
            return 0.5
        except Exception as e:
            logger.error(f"Ollama failed to generate score: {e}")
            return 0.5 # Return neutral score on failure
