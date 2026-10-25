
"""Надёжный клиент Ollama с ограниченной историей диалога."""

from collections import deque
from threading import Lock

import requests


class OllamaClient:
    def __init__(
        self,
        model="qwen2.5:7b",
        url="http://localhost:11434/api/chat",
        system_prompt=(
            "Ты JARVIS, полезный ассистент. "
            "Отвечай на русском языке, если пользователь не просит иначе. "
            "Не выдумывай факты и результаты действий. "
            "Если не знаешь ответа, честно скажи об этом."
        ),
        keep_alive="5m",
        max_history=12,
        timeout=120,
    ):
        self.model = model
        self.url = url
        self.system_prompt = system_prompt
        self.keep_alive = keep_alive
        self.max_history = max_history
        self.timeout = timeout

        self._history = deque(maxlen=max_history)
        self._lock = Lock()
        self.last_stats = {}

    @property
    def history(self):
        """Копия истории без возможности изменить внутреннее состояние."""
        with self._lock:
            return list(self._history)

    def clear_history(self):
        with self._lock:
            self._history.clear()

    def chat(self, user_text):
        user_text = user_text.strip()
        if not user_text:
            raise ValueError("Сообщение не должно быть пустым")

        # Блокировка не позволяет перемешать сообщения двух запросов.
        with self._lock:
            messages = [
                {"role": "system", "content": self.system_prompt},
                *list(self._history),
                {"role": "user", "content": user_text},
            ]

            payload = {
                "model": self.model,
                "messages": messages,
                "stream": False,
                "keep_alive": self.keep_alive,
                "options": {
                    "temperature": 0.2,
                    "num_ctx": 4096,
                    "num_predict": 256,
                },
            }

            response = requests.post(
                self.url,
                json=payload,
                timeout=self.timeout,
            )
            response.raise_for_status()
            data = response.json()

            answer = data.get("message", {}).get("content", "").strip()
            if not answer:
                raise RuntimeError("Ollama вернула пустой ответ")

            self._history.extend([
                {"role": "user", "content": user_text},
                {"role": "assistant", "content": answer},
            ])

            self.last_stats = {
                key: data[key]
                for key in (
                    "total_duration",
                    "load_duration",
                    "prompt_eval_count",
                    "eval_count",
                )
                if key in data
            }

            return answer
