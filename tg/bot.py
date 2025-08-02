import asyncio
import logging
import os
from dotenv import load_dotenv
from aiogram import Bot, Dispatcher, types
from aiogram.filters import Command
from aiogram.types import Message
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
# Импортируем наш ReAct агент
from app.agent.graph import create_graph, SYSTEM_PROMPT

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
        "Привет! Я ReAct агент на LangGraph. Могу помочь с прогнозом погоды! "
        "Спроси меня о погоде в любом городе."
    )

@dp.message(Command("help"))
async def cmd_help(message: Message):
    """Обработчик команды /help"""
    help_text = """
🤖 ReAct агент на LangGraph

Доступные команды:
/start - Начать работу с ботом
/help - Показать эту справку

Я могу помочь с прогнозом погоды! Примеры запросов:
• "Какая погода в Москве 2 августа 2025?"
• "Какая температура будет в Берлине?"
    """
    await message.answer(help_text)

@dp.message()
async def process_message(message: Message):
    """Обработчик всех текстовых сообщений - ReAct агент"""
    try:
        # Отправляем typing индикатор
        await bot.send_chat_action(message.chat.id, "typing")

        graph = create_graph()
        compiled_graph = graph.compile()

        # Обрабатываем сообщение с помощью ReAct агента
        user_input = message.text
        logger.info(f"Обработка сообщения от {message.from_user.id}: {user_input}")
        
        # Запускаем ReAct агента
       # response = await run_agent(user_input)
    
        response = await compiled_graph.ainvoke({
            "messages":
            [SystemMessage(content=SYSTEM_PROMPT),
            HumanMessage(content=user_input)]}
        )
        # Отправляем ответ
        response_text = response["messages"][-1].content
        await message.answer(response_text)
        
    except Exception as e:
        logger.error(f"Ошибка при обработке сообщения: {e}")
        await message.answer(
            "Извините, произошла ошибка при обработке вашего сообщения. "
            "Попробуйте еще раз или проверьте, что OpenAI API ключ настроен корректно."
        )

async def main():
    """Главная функция для запуска бота"""
    logger.info("Запуск бота...")
    
    # Запускаем поллинг
    await dp.start_polling(bot)

if __name__ == "__main__":
    # Запускаем бота
    asyncio.run(main()) 