#!/usr/bin/env python3
"""
Тестовый скрипт для проверки ReAct агента
"""

import asyncio
import os
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Импортируем наш агент
from app.agent.graph import run_agent

async def test_agent():
    """Тестирует ReAct агента"""
    
    # Проверяем наличие API ключа
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key == "your_openai_api_key_here":
        print("❌ OPENAI_API_KEY не настроен в .env файле")
        print("Добавьте реальный ключ OpenAI в .env файл")
        return
    
    print("🤖 Тестирование ReAct агента...")
    print("=" * 50)
    
    # Тестовые запросы
    test_queries = [
        "Какая погода в Москве 2 августа 2025?",
        "Какая температура будет в Берлине?",
        "Сравни погоду в Москве и Берлине"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\n{i}. Вопрос: {query}")
        print("-" * 30)
        
        try:
            response = await run_agent(query)
            print(f"Ответ: {response}")
        except Exception as e:
            print(f"❌ Ошибка: {e}")
    
    print("\n" + "=" * 50)
    print("✅ Тестирование завершено!")

if __name__ == "__main__":
    asyncio.run(test_agent())