import os
import time
from typing import Dict, List, Optional

from openai import OpenAI


class OpenAIClient:
    def __init__(
        self,
        model: str = 'gpt-4o',
        api_provider: str = 'openai',
        api_key: Optional[str] = None,
        api_base_url: Optional[str] = None,
        timeout_seconds: int = 120,
        medgemma_model: Optional[str] = None,
        medgemma_device: Optional[str] = None,
    ) -> None:
        provider = (api_provider or 'openai').strip().lower()

        if provider == 'medgemma':
            from src.agent_utils import MedGemmaClient
            mg_model = medgemma_model or 'google/medgemma-27b-text-it'
            mg_device = medgemma_device or 'cuda'
            self._client = MedGemmaClient(model_name=mg_model, device=mg_device)
            self._is_medgemma = True
        elif provider == 'openrouter':
            resolved_key = api_key or os.getenv('OPENROUTER_API_KEY')
            resolved_base_url = api_base_url or 'https://openrouter.ai/api/v1'
        else:
            resolved_key = api_key or os.getenv('OPENAI_API_KEY')
            resolved_base_url = api_base_url

        if not hasattr(self, '_is_medgemma'):
            self._is_medgemma = False

        if not self._is_medgemma:
            kwargs = {'api_key': resolved_key}
            if resolved_base_url:
                kwargs['base_url'] = resolved_base_url

            self._client = OpenAI(**kwargs).with_options(timeout=timeout_seconds)
        self.model = model
        self.api_provider = provider

    def call(self, messages: List[Dict[str, str]], max_retries: int = 3) -> str:
        attempt = 0
        # Reasoning models (gpt-5*, o*) reject non-default temperature, and their
        # reasoning tokens count against max_completion_tokens, so a small cap can
        # yield an empty message even on a successful call.
        is_reasoning = self.model.startswith(('gpt-5', 'o1', 'o3', 'o4'))
        params = {'max_completion_tokens': 8000} if is_reasoning else {
            'max_completion_tokens': 1000,
            'temperature': 1.25,
        }
        last_error = None

        while True:
            try:
                response = None

                if self._is_medgemma:
                    response = self._client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        max_tokens=1000,
                    )
                else:
                    while True:
                        try:
                            response = self._client.chat.completions.create(
                                model=self.model,
                                messages=messages,
                                **params,
                            )
                            break
                        except Exception as first_error:
                            # Drop whichever parameter the API rejects and retry once per parameter.
                            first_error_text = str(first_error).lower()
                            dropped = False
                            for bad_param, replacement in [
                                ('max_completion_tokens', {'max_tokens': params.get('max_completion_tokens', 1000)}),
                                ('temperature', {}),
                                ('max_tokens', {}),
                            ]:
                                if bad_param in params and bad_param in first_error_text and (
                                    'unsupported' in first_error_text or 'does not support' in first_error_text
                                ):
                                    del params[bad_param]
                                    params.update(replacement)
                                    dropped = True
                                    break
                            if not dropped:
                                raise

                content = response.choices[0].message.content if response and response.choices else ""
                content = (content or "").strip()
                if not content:
                    raise RuntimeError(
                        f"Empty completion from {self.model} "
                        f"(finish_reason={response.choices[0].finish_reason if response and response.choices else 'n/a'})"
                    )
                return content
            except Exception as e:
                last_error = e
                attempt += 1
                if attempt > max_retries:
                    print(f"[OpenAIClient] call failed after {max_retries} retries: {last_error}", flush=True)
                    return ""

                # Exponential backoff: 4s, 16s, 64s ...
                sleep_s = 4 ** attempt
                time.sleep(sleep_s)
                continue
