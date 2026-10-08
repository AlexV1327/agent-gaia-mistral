import os
import random
import time

from dotenv import load_dotenv
from mistralai.client import Mistral

load_dotenv()


class MistralLLM:
    def __init__(self, model: str | None = None):
        api_key = os.getenv("MISTRAL_API_KEY")
        if not api_key:
            raise ValueError("MISTRAL_API_KEY introuvable dans .env")

        self.client = Mistral(api_key=api_key)
        self.model = model or os.getenv("MISTRAL_MODEL") or "mistral-small-latest"

    def _is_retryable_error(self, error: Exception) -> bool:
        text = str(error).lower()

        retryable_markers = [
            "429",
            "rate limit",
            "rate_limited",
            "service_tier_capacity_exceeded",
            "timeout",
            "timed out",
            "temporarily unavailable",
            "connection",
            "502",
            "503",
            "504",
        ]

        return any(marker in text for marker in retryable_markers)

    def _sleep_before_retry(self, attempt: int, error: Exception) -> None:
        text = str(error).lower()

        if "429" in text or "rate" in text or "capacity" in text:
            base_wait = 10
        else:
            base_wait = 3

        wait_time = min(60, base_wait * attempt)
        jitter = random.uniform(0, 2)

        total_wait = wait_time + jitter
        print(f"Nouvelle tentative dans {total_wait:.1f} secondes...")
        time.sleep(total_wait)

    def chat_with_tools(self, messages: list[dict], tools: list[dict], max_retries: int = 4):
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.complete(
                    model=self.model,
                    messages=messages,
                    tools=tools,
                    tool_choice="auto",
                )
                return response
            except Exception as e:
                last_error = e
                print(f"[Tentative {attempt}/{max_retries}] Erreur API Mistral : {e}")

                if attempt >= max_retries or not self._is_retryable_error(e):
                    break

                self._sleep_before_retry(attempt, e)

        raise RuntimeError(f"Échec après {max_retries} tentatives : {last_error}")

    def chat(self, messages: list[dict], max_retries: int = 4):
        last_error = None

        for attempt in range(1, max_retries + 1):
            try:
                response = self.client.chat.complete(
                    model=self.model,
                    messages=messages,
                )
                return response
            except Exception as e:
                last_error = e
                print(f"[Tentative {attempt}/{max_retries}] Erreur API Mistral : {e}")

                if attempt >= max_retries or not self._is_retryable_error(e):
                    break

                self._sleep_before_retry(attempt, e)

        raise RuntimeError(f"Échec après {max_retries} tentatives : {last_error}")
