import asyncio
import logging
import os
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message

# Загрузка переменных окружения из .env файла
load_dotenv()

# Настройка логирования
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Получение токена из переменных окружения
BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")

if not BOT_TOKEN:
    raise ValueError("TELEGRAM_BOT_TOKEN не найден в переменных окружения")

# Инициализация бота и диспетчера
bot = Bot(token=BOT_TOKEN)
dp = Dispatcher()

@dp.message(Command("start"))
async def cmd_start(message: Message):
    """Обработчик команды /start"""
    await message.answer(
        "Привет! Я эхо-бот. Отправь мне любое сообщение, и я повторю его."
    )

@dp.message(Command("help"))
async def cmd_help(message: Message):
    """Обработчик команды /help"""
    help_text = """
🤖 Эхо-бот

Доступные команды:
/start - Начать работу с ботом
/help - Показать эту справку

Просто напиши мне любое сообщение, и я повторю его!
    """
    await message.answer(help_text)

@dp.message()
async def echo_message(message: Message):
    """Обработчик всех текстовых сообщений - echo функциональность"""
    try:
        # Отправляем typing индикатор
        await bot.send_chat_action(message.chat.id, "typing")
        
        # Повторяем сообщение пользователя
        await message.answer(f"Echo: {message.text}")
        
    except Exception as e:
        logger.error(f"Ошибка при обработке сообщения: {e}")
        await message.answer("Произошла ошибка при обработке сообщения.")

async def main():
    """Главная функция для запуска бота"""
    logger.info("Запуск бота...")
    
    # Запускаем поллинг
    await dp.start_polling(bot)

if __name__ == "__main__":
    # Запускаем бота
    asyncio.run(main()) 