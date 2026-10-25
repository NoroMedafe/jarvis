import json
import subprocess
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import whisper
import requests
import os
import threading
import webbrowser
import torch
import customtkinter as ctk
from urllib.parse import quote

# ================= НАСТРОЙКИ =================
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:7b"

# Глобальные переменные
whisper_model = None
tts_model = None
tts_sample_rate = 48000
tts_lock = threading.Lock() # Блокировка, чтобы фразы не накладывались

# Словарь приложений
APPS = {
    "браузер": "browser",
    "хром": "browser",
    "дискорд": "discord",
    "телеграм": "telegram",
    "блокнот": "notepad",
    "калькулятор": "calc",
    "проводник": "explorer",
    "vscode": "vscode",
    "steam": "steam",
    "spotify": "spotify"
}

APP_PATHS = {
    "discord": r"C:\Users\danya\AppData\Local\Discord\Update.exe --processStart Discord.exe",
    "telegram": r"C:\Users\danya\AppData\Roaming\Telegram Desktop\Telegram.exe",
    "spotify": r"C:\Users\danya\AppData\Roaming\Spotify\Spotify.exe",
    "vscode": r"C:\Users\danya\AppData\Local\Programs\Microsoft VS Code\Code.exe",
    "steam": r"C:\Program Files (x86)\Steam\steam.exe",
}


def get_weather(city="Санкт-Петербург"):
    try:
        url = f"https://wttr.in/{quote(city)}?format=j1&lang=ru"
        response = requests.get(url, timeout=10)
        data = response.json()
        current = data["current_condition"][0]
        temp = current["temp_C"]
        feels_like = current["FeelsLikeC"]
        humidity = current["humidity"]
        weather_ru = current["lang_ru"][0]["value"] if "lang_ru" in current else current["weatherDesc"][0]["value"]
        return f"В городе {city} сейчас {temp} градусов, ощущается как {feels_like}. {weather_ru}. Влажность {humidity}%."
    except Exception as e:
        return f"Не удалось получить погоду: {e}"


def search_web(query):
    webbrowser.open(f"https://duckduckgo.com/?q={quote(query)}")
    return f"Ищу '{query}' в интернете."


def speak_text(text):
    """Локальная, мгновенная озвучка через Silero TTS"""
    if not text or tts_model is None:
        return
    
    with tts_lock:
        try:
            print(f"🔊 Озвучиваю: {text}")
            # Используем актуальный спикер v4_ru
            audio = tts_model.apply_tts(text=text, speaker='v4_ru', sample_rate=tts_sample_rate)
            
            # Преобразуем тензор PyTorch в numpy массив для sounddevice
            audio_np = audio.numpy()
            
            # Воспроизводим звук
            sd.play(audio_np, tts_sample_rate)
            sd.wait() # Ждем окончания воспроизведения
        except Exception as e:
            print(f"❌ Ошибка TTS: {e}")


class JarvisApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Джарвис")
        self.geometry("500x600")
        self.resizable(False, False)

        self.is_recording = False
        self.audio_data = []
        self.sample_rate = 16000
        self.stream = None

        self.status_var = ctk.StringVar(value="⏳ Загрузка моделей...")
        self.user_text_var = ctk.StringVar(value="")
        self.jarvis_text_var = ctk.StringVar(value="")

        self.create_widgets()
        print("🔧 Запускаю поток инициализации...")
        threading.Thread(target=self.init_models, daemon=True).start()

    def init_models(self):
        global whisper_model, tts_model, tts_sample_rate
        try:
            print("⏳ Загружаю Whisper...")
            whisper_model = whisper.load_model("medium")
            print("✅ Whisper загружен")

            print("⏳ Загружаю Silero TTS (локальный голос, ~30 сек)...")
            tts_model, tts_sample_rate = torch.hub.load(
                repo_or_dir='snakers4/silero-models',
                model='silero_tts',
                language='ru',
                speaker='v4_ru', # Исправлено на актуальный спикер из списка
                trust_repo=True
            )
            tts_model.to('cpu') 
            print("✅ Silero TTS готов")

            self.after(0, lambda: self.status_var.set("✅ Готов! Зажми кнопку и говори"))
            print("✅ Инициализация завершена")
        except Exception as err:
            err_msg = str(err)
            print(f"❌ ОШИБКА В init_models: {err_msg}")
            import traceback
            traceback.print_exc()
            self.after(0, lambda msg=err_msg: self.status_var.set(f"❌ Ошибка: {msg}"))

    def create_widgets(self):
        title = ctk.CTkLabel(self, text="ДЖАРВИС", font=ctk.CTkFont(size=28, weight="bold"))
        title.pack(pady=20)

        ctk.CTkLabel(self, textvariable=self.status_var, font=ctk.CTkFont(size=14)).pack(pady=10)

        self.mic_button = ctk.CTkButton(
            self,
            text="🎤 НАЖМИ И ГОВОРИ",
            font=ctk.CTkFont(size=20, weight="bold"),
            width=300,
            height=150,
            corner_radius=75,
            fg_color="#2b6cb0",
            hover_color="#1e4e8c"
        )
        self.mic_button.pack(pady=30)
        self.mic_button.bind("<ButtonPress-1>", self.start_recording)
        self.mic_button.bind("<ButtonRelease-1>", self.stop_recording)

        user_frame = ctk.CTkFrame(self)
        user_frame.pack(pady=10, padx=20, fill="x")
        ctk.CTkLabel(user_frame, text="Ты сказал:", font=ctk.CTkFont(size=12)).pack(anchor="w", padx=10, pady=5)
        ctk.CTkLabel(user_frame, textvariable=self.user_text_var, font=ctk.CTkFont(size=14), wraplength=450).pack(anchor="w", padx=10, pady=5)

        jarvis_frame = ctk.CTkFrame(self)
        jarvis_frame.pack(pady=10, padx=20, fill="x")
        ctk.CTkLabel(jarvis_frame, text="Джарвис:", font=ctk.CTkFont(size=12)).pack(anchor="w", padx=10, pady=5)
        ctk.CTkLabel(jarvis_frame, textvariable=self.jarvis_text_var, font=ctk.CTkFont(size=14), wraplength=450).pack(anchor="w", padx=10, pady=5)

    def start_recording(self, event):
        if whisper_model is None:
            self.status_var.set("⏳ Модели ещё загружаются...")
            return

        self.is_recording = True
        self.audio_data = []
        self.status_var.set("🎤 Слушаю...")
        self.user_text_var.set("")
        self.jarvis_text_var.set("")

        threading.Thread(target=self.record_audio, daemon=True).start()

    def record_audio(self):
        def callback(indata, frames, time, status):
            if self.is_recording:
                self.audio_data.append(indata.copy())

        try:
            self.stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype='int16',
                callback=callback
            )
            self.stream.start()
            while self.is_recording:
                sd.sleep(100)
        except Exception as e:
            self.after(0, lambda err=str(e): self.status_var.set(f"❌ Ошибка микрофона: {err}"))

    def stop_recording(self, event):
        if not self.is_recording:
            return

        self.is_recording = False

        if self.stream:
            self.stream.stop()
            self.stream.close()

        if len(self.audio_data) == 0:
            self.status_var.set("⚠️ Аудио не записано")
            return

        threading.Thread(target=self.process_audio, daemon=True).start()

    def process_audio(self):
        self.after(0, lambda: self.status_var.set("🔍 Распознаю речь..."))

        audio = np.concatenate(self.audio_data, axis=0)
        wavfile.write("temp_audio.wav", self.sample_rate, audio)

        result = whisper_model.transcribe("temp_audio.wav", language="ru", fp16=True)
        text = result["text"].strip()

        self.after(0, lambda: self.user_text_var.set(text))

        if not text:
            self.after(0, lambda: self.status_var.set("⚠️ Не услышал"))
            return

        self.after(0, lambda: self.status_var.set("🧠 Думаю..."))
        action_data = self.ask_llm(text)
        result_text = self.execute_action(action_data)

        self.after(0, lambda: self.jarvis_text_var.set(result_text))
        self.after(0, lambda: self.status_var.set("✅ Готов! Зажми кнопку и говори"))
        
        # Озвучиваем в фоновом потоке (не блокирует интерфейс)
        threading.Thread(target=speak_text, args=(result_text,), daemon=True).start()

    def ask_llm(self, user_text):
        system_prompt = """
        Ты - Джарвис, голосовой помощник на русском языке.
        
        ВАЖНО: Отвечай ТОЛЬКО на русском языке. Никогда не используй китайские иероглифы или другие языки.
        Отвечай СТРОГО в формате JSON без markdown, без пояснений, без ```json.
        
        Доступные действия:
        1. {"action": "open_app", "app_name": "название"} - открыть приложение.
           Список: браузер, хром, дискорд, телеграм, блокнот, калькулятор, проводник, vscode, steam, spotify.
        2. {"action": "get_weather", "city": "город"} - узнать погоду. Если город не назван, используй "Санкт-Петербург".
        3. {"action": "search_web", "query": "запрос"} - найти в интернете и открыть браузер.
        4. {"action": "answer", "text": "ответ"} - просто ответить голосом.
        """

        payload = {
            "model": MODEL_NAME,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            "format": "json",
            "keep_alive": 0,
            "stream": False
        }

        try:
            response = requests.post(OLLAMA_URL, json=payload, timeout=60)
            response.raise_for_status()
            result = response.json()
            llm_response = result['message']['content']
            llm_response = llm_response.replace("```json", "").replace("```", "").strip()
            return json.loads(llm_response)
        except Exception as err:
            print(f"❌ Ошибка LLM: {err}")
            return {"action": "answer", "text": "Прости, я не смог обработать запрос."}

    def execute_action(self, action_data):
        action = action_data.get("action")

        if action == "open_app":
            app_key = action_data.get("app_name", "").lower()
            
            if app_key in ("браузер", "хром", "browser", "chrome"):
                webbrowser.open("https://google.com")
                return "Открываю браузер."
            
            if app_key in APP_PATHS:
                try:
                    subprocess.Popen(APP_PATHS[app_key], shell=True)
                    return f"Запускаю {app_key}."
                except Exception as e:
                    print(f"Ошибка запуска {app_key}: {e}")
                    return f"Не удалось открыть {app_key}."
            
            if app_key in APPS:
                try:
                    subprocess.Popen(APPS[app_key], shell=True)
                    return f"Запускаю {app_key}."
                except Exception as e:
                    print(f"Ошибка запуска {app_key}: {e}")
                    return f"Не удалось открыть {app_key}."
            
            return f"Я не знаю приложение {app_key}."

        elif action == "get_weather":
            city = action_data.get("city", "Санкт-Петербург")
            return get_weather(city)

        elif action == "search_web":
            query = action_data.get("query", "")
            return search_web(query)

        elif action == "answer":
            return action_data.get("text", "Хорошо.")

        return "Я не понял команду."


if __name__ == "__main__":
    try:
        print("🚀 Запуск Джарвиса...")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        print("🎨 Тема установлена")

        app = JarvisApp()
        print("🖼️ Приложение создано, запускаем цикл...")
        app.mainloop()
    except Exception as err:
        print(f"💥 КРИТИЧЕСКАЯ ОШИБКА: {err}")
        import traceback
        traceback.print_exc()
        input("Нажми Enter для выхода...")