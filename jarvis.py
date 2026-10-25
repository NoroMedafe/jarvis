import json
import subprocess
import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wavfile
import whisper
import pyttsx3
import requests
import os

# ================= НАСТРОЙКИ =================
OLLAMA_URL = "http://localhost:11434/api/chat"
MODEL_NAME = "qwen2.5:7b"
RECORD_SECONDS = 6  # Время записи после нажатия Enter

# Словарь приложений (можешь добавлять свои)
APPS = {
    "браузер": "chrome",
    "хром": "chrome",
    "дискорд": "discord",
    "телеграм": "Telegram",
    "блокнот": "notepad",
    "калькулятор": "calc",
    "проводник": "explorer",
    "vscode": "code",
    "steam": "steam"
}

# ================= ИНИЦИАЛИЗАЦИЯ =================
print("⏳ Загрузка моделей (при первом запуске это займет пару минут)...")

# Используем модель 'medium' для отличного русского. 
# Если будет тормозить, поменяй на 'small' или 'base'.
whisper_model = whisper.load_model("medium") 

tts_engine = pyttsx3.init()
tts_engine.setProperty('rate', 160)  # Скорость речи (чуть быстрее обычной)
tts_engine.setProperty('volume', 1.0) 
print("✅ Джарвис готов! Нажми Enter, чтобы сказать команду.")

# ================= ФУНКЦИИ =================

def record_audio():
    """Запись звука с микрофона"""
    print("🎤 Слушаю...")
    sample_rate = 16000
    audio = sd.rec(int(sample_rate * RECORD_SECONDS), samplerate=sample_rate, channels=1, dtype='int16')
    sd.wait()
    print("🛑 Запись завершена. Думаю...")
    return audio, sample_rate

def transcribe_audio(audio, sample_rate):
    """Распознавание речи через openai-whisper"""
    wavfile.write("temp_audio.wav", sample_rate, audio)
    
    # Распознавание
    result = whisper_model.transcribe("temp_audio.wav", language="ru", fp16=True)
    text = result["text"].strip()
    print(f"🗣 Ты сказал: \"{text}\"")
    return text

def ask_llm(user_text):
    """Отправка запроса в Ollama"""
    system_prompt = """
    Ты - Джарвис, голосовой помощник. Отвечай СТРОГО в формате JSON без markdown и лишнего текста.
    Доступные действия:
    1. {"action": "open_app", "app_name": "название_из_списка"} - открыть приложение.
    2. {"action": "answer", "text": "твой ответ"} - просто ответить голосом.
    
    Список приложений: браузер, хром, дискорд, телеграм, блокнот, калькулятор, проводник, vscode, steam.
    """

    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_text}
        ],
        "format": "json",       # Принудительный вывод JSON
        "keep_alive": 0,        # Выгружает модель из VRAM сразу после ответа!
        "stream": False
    }

    try:
        response = requests.post(OLLAMA_URL, json=payload, timeout=60)
        response.raise_for_status()
        result = response.json()
        llm_response = result['message']['content']
        
        # Иногда LLM оборачивает JSON в ```json ... ```, чистим это
        llm_response = llm_response.replace("```json", "").replace("```", "").strip()
        
        return json.loads(llm_response)
    except Exception as e:
        print(f"❌ Ошибка LLM: {e}")
        return {"action": "answer", "text": "Прости, я не смог обработать запрос."}

def execute_action(action_data):
    """Выполнение команды"""
    action = action_data.get("action")
    
    if action == "open_app":
        app_key = action_data.get("app_name", "").lower()
        app_to_run = APPS.get(app_key)
        
        if app_to_run:
            try:
                subprocess.Popen(app_to_run, shell=True)
                return f"Запускаю {app_key}."
            except Exception:
                return f"Не удалось открыть {app_key}. Проверь название."
        else:
            return f"Я не знаю приложение {app_key}. Вот что я умею: {', '.join(APPS.keys())}."
            
    elif action == "answer":
        return action_data.get("text", "Хорошо.")
        
    return "Я не понял команду."

def speak(text):
    """Озвучка ответа"""
    print(f"🤖 Джарвис: {text}")
    tts_engine.say(text)
    tts_engine.runAndWait()

# ================= ГЛАВНЫЙ ЦИКЛ =================

def main():
    if os.path.exists("temp_audio.wav"):
        os.remove("temp_audio.wav")

    while True:
        try:
            input("\n[Нажми Enter, чтобы говорить, или Ctrl+C для выхода]")
            
            audio, sr = record_audio()
            text = transcribe_audio(audio, sr)
            
            if not text:
                speak("Я тебя не услышал.")
                continue
                
            action_data = ask_llm(text)
            print(f"⚙️ Решение модели: {action_data}")
            
            result_text = execute_action(action_data)
            speak(result_text)
            
        except KeyboardInterrupt:
            print("\n👋 Джарвис отключается.")
            break
        except Exception as e:
            print(f"⚠️ Общая ошибка: {e}")

if __name__ == "__main__":
    main()