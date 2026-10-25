
import unittest
from unittest.mock import Mock, patch

import requests

from llm_client import OllamaClient


class TestOllamaClient(unittest.TestCase):
    def setUp(self):
        self.client = OllamaClient(max_history=4)

    @patch("llm_client.requests.post")
    def test_chat_sends_system_prompt_and_returns_answer(self, post):
        response = Mock()
        response.json.return_value = {
            "message": {"content": "Привет!"},
            "total_duration": 100,
            "load_duration": 10,
        }
        post.return_value = response

        answer = self.client.chat("Привет")
        self.assertEqual(answer, "Привет!")

        payload = post.call_args.kwargs["json"]
        self.assertEqual(payload["model"], "qwen2.5:7b")
        self.assertEqual(payload["messages"][0]["role"], "system")
        self.assertEqual(payload["messages"][-1]["content"], "Привет")
        self.assertEqual(payload["keep_alive"], "30m")
        self.assertFalse(payload["stream"])

    @patch("llm_client.requests.post")
    def test_next_message_contains_previous_turn(self, post):
        response = Mock()
        response.json.side_effect = [
            {"message": {"content": "Меня зовут JARVIS."}},
            {"message": {"content": "Ты назвал меня JARVIS."}},
        ]
        post.return_value = response

        self.client.chat("Как тебя зовут?")
        answer = self.client.chat("Что я спросил раньше?")

        second_payload = post.call_args.kwargs["json"]
        messages = second_payload["messages"]

        self.assertEqual(
            messages[-3]["content"],
            "Как тебя зовут?",
        )
        self.assertEqual(
            messages[-2]["content"],
            "Меня зовут JARVIS.",
        )
        self.assertEqual(
            answer,
            "Ты назвал меня JARVIS.",
        )

    @patch("llm_client.requests.post")
    def test_history_can_be_cleared(self, post):
        response = Mock()
        response.json.return_value = {
            "message": {"content": "Ответ"}
        }
        post.return_value = response

        self.client.chat("Первый вопрос")
        self.client.clear_history()

        self.assertEqual(self.client.history, [])

    def test_empty_message_is_rejected(self):
        with self.assertRaises(ValueError):
            self.client.chat("   ")

    @patch("llm_client.requests.post")
    def test_http_error_is_propagated(self, post):
        response = Mock()
        response.raise_for_status.side_effect = (
            requests.HTTPError("500 Server Error")
        )
        post.return_value = response

        with self.assertRaises(requests.HTTPError):
            self.client.chat("Привет")

        self.assertEqual(self.client.history, [])

    @patch("llm_client.requests.post")
    def test_empty_response_is_rejected(self, post):
        response = Mock()
        response.json.return_value = {
            "message": {"content": "  "}
        }
        post.return_value = response

        with self.assertRaises(RuntimeError):
            self.client.chat("Привет")

        self.assertEqual(self.client.history, [])


if __name__ == "__main__":
    unittest.main()
