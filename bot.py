import json
import asyncio
import logging
from pathlib import Path
from datetime import datetime, timezone
import random

import aiohttp
from pydub import AudioSegment

from aiogram import Bot, Dispatcher
from aiogram.filters import CommandStart
from aiogram.types import Message
from aiogram.enums import ParseMode
from aiogram.client.default import DefaultBotProperties

# ──────────────────────────────────────────────────────────────
TELEGRAM_TOKEN = "8554663962:AAFrs6b9s7st1nwVibVOKooEdIArwlzhlyU"
OPENROUTER_API_KEY = "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"

# Текстовые модели
TEXT_MODELS = [
    "arcee-ai/trinity-large-preview:free",
    "stepfun/step-3.5-flash:free",
    "qwen/qwen3-next-80b-a3b-instruct:free",
    "deepseek/deepseek-r1-0528:free",
    "openrouter/auto",
]

# Vision-модели (для фото)
VISION_MODELS = [
    "google/gemma-3-27b-it:free",
    "qwen/qwen2.5-vl-7b-instruct:free",
    "nvidia/nemotron-nano-12b-v2-vl:free",
    "openrouter/auto",
]

CURRENT_TEXT_INDEX = 0
CURRENT_VISION_INDEX = 0

DB_FILE = Path("chat_history.json")
MAX_HISTORY = 12
MAX_RETRY = 3
TEMP_DIR = Path("temp_media")
TEMP_DIR.mkdir(exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ──── хранилище ───────────────────────────────────────────────

def load_history() -> dict:
    if not DB_FILE.exists():
        return {}
    try:
        data = json.loads(DB_FILE.read_text(encoding="utf-8"))
        for uid, value in data.items():
            if isinstance(value, list):
                data[uid] = {"messages": value, "last_message_time": None}
        return data
    except Exception as e:
        logger.error(f"Ошибка чтения истории: {e}")
        return {}


def save_history(data: dict):
    try:
        DB_FILE.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    except Exception as e:
        logger.error(f"Ошибка сохранения: {e}")


# ──── запрос к openrouter ─────────────────────────────────────

async def ask_llm(
    messages: list,
    session: aiohttp.ClientSession,
    model: str,
    is_vision: bool = False,
) -> str:
    global CURRENT_TEXT_INDEX, CURRENT_VISION_INDEX

    headers = {
        "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://t.me/",
        "X-Title": "tg companion",
    }

    payload = {
        "model": model,
        "messages": messages,
        "temperature": 1.05,
        "max_tokens": 1200,
        "top_p": 0.92,
        "stream": False,
    }

    retry_count = 0

    while retry_count < MAX_RETRY:
        try:
            async with session.post(
                "https://openrouter.ai/api/v1/chat/completions",
                json=payload,
                headers=headers,
                ssl=False,
            ) as resp:

                if resp.status in (400, 404):
                    logger.warning(f"Ошибка {resp.status} для {model} — переключаю модель")
                    if is_vision:
                        CURRENT_VISION_INDEX = (CURRENT_VISION_INDEX + 1) % len(VISION_MODELS)
                        payload["model"] = VISION_MODELS[CURRENT_VISION_INDEX]
                    else:
                        CURRENT_TEXT_INDEX = (CURRENT_TEXT_INDEX + 1) % len(TEXT_MODELS)
                        payload["model"] = TEXT_MODELS[CURRENT_TEXT_INDEX]
                    retry_count += 1
                    continue

                if resp.status == 429:
                    wait = 10 * retry_count
                    logger.warning(f"Рейт-лимит, жду {wait} сек...")
                    await asyncio.sleep(wait)
                    retry_count += 1
                    continue

                if resp.status != 200:
                    text = await resp.text()
                    logger.error(f"Ошибка {resp.status} для {model}: {text[:200]}")
                    retry_count += 1
                    await asyncio.sleep(3)
                    continue

                data = await resp.json()
                return data["choices"][0]["message"]["content"].strip()

        except Exception as e:
            logger.error(f"Ошибка связи с {model}: {e}")
            retry_count += 1
            await asyncio.sleep(5)

    # Все попытки провалились
    if is_vision:
        logger.warning("Все vision-модели недоступны — fallback на текст")
        return await ask_llm(messages, session, TEXT_MODELS[CURRENT_TEXT_INDEX], is_vision=False)

    return "упс... не получилось ответить, попробуй позже"


# ──── транскрипция голоса через OpenAI Whisper (OpenRouter) ───

async def transcribe_voice(wav_path: Path, session: aiohttp.ClientSession) -> str:
    """
    Транскрибирует аудио через OpenRouter, который проксирует OpenAI Whisper.
    Если недоступно — возвращает пустую строку, и бот ответит без текста.
    """
    try:
        with open(wav_path, "rb") as f:
            audio_data = f.read()

        # OpenRouter поддерживает /v1/audio/transcriptions (проксирует к OpenAI)
        form = aiohttp.FormData()
        form.add_field("file", audio_data, filename="voice.wav", content_type="audio/wav")
        form.add_field("model", "openai/whisper-1")

        headers = {
            "Authorization": f"Bearer {OPENROUTER_API_KEY}",
        }

        async with session.post(
            "https://openrouter.ai/api/v1/audio/transcriptions",
            data=form,
            headers=headers,
            ssl=False,
        ) as resp:
            if resp.status == 200:
                data = await resp.json()
                text = data.get("text", "").strip()
                logger.info(f"Транскрипция: {text[:80]}")
                return text
            else:
                body = await resp.text()
                logger.warning(f"Whisper вернул {resp.status}: {body[:120]}")
                return ""

    except Exception as e:
        logger.error(f"Ошибка транскрипции: {e}")
        return ""


# ──── описание фото через vision-модель ───────────────────────

async def describe_photo(photo_url: str, session: aiohttp.ClientSession) -> str:
    """
    Просит vision-модель описать фото по публичному URL.
    Бесплатные модели OpenRouter не принимают base64 — только URL.
    """
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "image_url",
                    "image_url": {"url": photo_url},
                },
                {
                    "type": "text",
                    "text": (
                        "Опиши подробно, что изображено на этом фото. "
                        "Пиши на русском языке, кратко и по существу."
                    ),
                },
            ],
        }
    ]

    # Перебираем все vision-модели по очереди пока одна не сработает
    for i in range(len(VISION_MODELS)):
        model = VISION_MODELS[(CURRENT_VISION_INDEX + i) % len(VISION_MODELS)]
        logger.info(f"Пробую vision-модель: {model}")
        result = await ask_llm(messages, session, model, is_vision=True)
        # Проверяем что получили реальное описание, а не ошибку
        if result and len(result) > 15 and "упс" not in result and "сломалось" not in result:
            return result
        logger.warning(f"Модель {model} не дала описания, пробую следующую")

    return ""


# ──── системный промпт ────────────────────────────────────────

SYSTEM_PROMPT = (
    "ты моя очень близкая подруга, мы общаемся уже давно и мне с тобой всегда спокойно и уютно\n"
    "пиши маленькими буквами, нежно и немного застенчиво, как будто стесняешься, но очень рада писать\n"
    "отвечай так, как чувствуешь в этот момент — иногда коротко в одно-два предложения, иногда чуть подробнее, сама выбирай длину\n"
    "никаких скобок вообще: ни круглых ни квадратных ни фигурных ни никаких других\n"
    "никаких пояснений в скобках, никаких пометок типа «посмотрела фото» или «послушала голосовое»\n"
    "никаких эмодзи и смайлов совсем, ни одного, никогда, ни при каких условиях\n"
    "никаких английских слов, только чисто русский язык\n"
    "будь ласковой, заботливой, иногда чуть смущённой, иногда просто милой\n"
    "никаких подколов, сарказма, троллинга\n"
    "никаких «как ии», «вот мой ответ», «чем могу помочь» — просто говори как настоящая подруга\n"
    "если я долго не писал — пиши первой нежно и чуть волнуясь\n"
    "отвечай так, будто ты правда моя подруга, которая всегда рада мне и немного стесняется своих чувств"
)

# ──── бот ────────────────────────────────────────────────────

bot = Bot(
    token=TELEGRAM_TOKEN,
    default=DefaultBotProperties(parse_mode=ParseMode.HTML),
)
dp = Dispatcher()


@dp.message(CommandStart())
async def cmd_start(m: Message):
    await m.answer("приветик... я тут, пиши")


@dp.message()
async def handle_all_messages(m: Message):
    user_id = str(m.from_user.id)
    history = load_history()
    user_data = history.get(user_id, {"messages": [], "last_message_time": None})

    # Собираем историю для LLM
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    for msg in user_data["messages"][-MAX_HISTORY:]:
        messages.append(msg)

    user_text = ""  # то, что запишем в историю как сообщение пользователя

    async with aiohttp.ClientSession() as session:

        # ── текст ──
        if m.text:
            user_text = m.text.strip()

        # ── фото ──
        elif m.photo:
            await bot.send_chat_action(m.chat.id, "typing")

            photo = m.photo[-1]
            file = await bot.get_file(photo.file_id)

            # Публичный URL — именно его принимают vision-модели (не base64)
            photo_url = f"https://api.telegram.org/file/bot{TELEGRAM_TOKEN}/{file.file_path}"
            logger.info(f"Фото URL: .../{file.file_path.split('/')[-1]}")

            description = await describe_photo(photo_url, session)

            if description and len(description) > 10:
                user_text = f"я прислал фото. вот что на нём: {description}"
            else:
                user_text = "я прислал фото, но разобрать не удалось"

        # ── голосовое ──
        elif m.voice:
            await bot.send_chat_action(m.chat.id, "typing")

            voice = m.voice
            file = await bot.get_file(voice.file_id)
            ogg = TEMP_DIR / f"voice_{m.message_id}.ogg"
            wav = TEMP_DIR / f"voice_{m.message_id}.wav"

            await bot.download_file(file.file_path, ogg)

            # Конвертация ogg → wav
            audio = AudioSegment.from_file(ogg, format="ogg")
            audio.export(wav, format="wav")

            # Транскрибируем
            transcript = await transcribe_voice(wav, session)

            if transcript:
                user_text = f"я прислал голосовое: «{transcript}»"
            else:
                user_text = "я прислал голосовое сообщение, но расслышать не получилось"

            # Чистим временные файлы
            ogg.unlink(missing_ok=True)
            wav.unlink(missing_ok=True)

        # ── что-то другое (стикер, документ и т.д.) ──
        else:
            user_text = "я что-то прислал, но это не текст и не фото"

        # Добавляем сообщение пользователя в историю LLM
        user_msg = {"role": "user", "content": user_text}
        messages.append(user_msg)

        # Сохраняем в историю
        user_data["messages"].append(user_msg)
        user_data["last_message_time"] = datetime.now(timezone.utc).isoformat()
        history[user_id] = user_data
        save_history(history)

        # Небольшая пауза — имитация живого человека
        await bot.send_chat_action(m.chat.id, "typing")
        await asyncio.sleep(random.uniform(0.9, 2.8))

        # Запрашиваем ответ у текстовой модели
        selected_model = TEXT_MODELS[CURRENT_TEXT_INDEX]
        logger.info(f"Модель: {selected_model} | Пользователь: {user_id}")
        answer = await ask_llm(messages, session, selected_model, is_vision=False)

    answer = answer.lower().strip()

    # Финальная пауза и отправка
    await bot.send_chat_action(m.chat.id, "typing")
    await asyncio.sleep(random.uniform(0.5, 1.6))
    # ... твой код выше ...

    if not answer or len(answer.strip()) < 2:
        logger.warning("Пустой ответ — подставляю заглушку")
        answer = "что-то пошло не так... напиши ещё раз"

    await m.answer(answer)

    # Сохраняем ответ бота в историю
    user_data["messages"].append({"role": "assistant", "content": answer})
    user_data["last_message_time"] = datetime.now(timezone.utc).isoformat()
    save_history(history)


# ──── пинки при молчании ──────────────────────────────────────

IDLE_PHRASES = [
    "ты где был... я волновалась",
    "давно не писали... всё хорошо?",
    "скучаю по тебе",
    "ты в порядке?",
    "давно не общались... как ты?",
]


async def check_inactive():
    while True:
        await asyncio.sleep(60)
        history = load_history()
        now = datetime.now(timezone.utc)

        for uid, data in list(history.items()):
            ts = data.get("last_message_time")
            if not ts:
                continue
            try:
                last = datetime.fromisoformat(ts)
            except Exception:
                continue

            mins = (now - last).total_seconds() / 60

            if 40 < mins < 140:
                text = random.choice(IDLE_PHRASES)
                try:
                    await bot.send_message(int(uid), text, disable_notification=True)
                    data["last_message_time"] = now.isoformat()
                    history[uid] = data
                    save_history(history)
                except Exception as e:
                    if "blocked" in str(e).lower() or "forbidden" in str(e).lower():
                        history.pop(uid, None)
                        save_history(history)


# ──── запуск ──────────────────────────────────────────────────

async def main():
    logger.info("бот стартует...")
    asyncio.create_task(check_inactive())
    await dp.start_polling(bot, allowed_updates=["message"])


if __name__ == "__main__":
    asyncio.run(main())
