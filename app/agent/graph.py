"""
ReAct Agent implementation from scratch with LangGraph
Based on: https://www.philschmid.de/langgraph-gemini-2-5-react-agent
"""

import os
from typing import Annotated, TypedDict, List
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage, ToolMessage, add_messages
from langchain_core.tools import tool
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from langchain_openai import ChatOpenAI
from dotenv import load_dotenv

# Загружаем переменные окружения
load_dotenv()

# Определяем состояние агента
class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]

# Создаем инструмент для получения погоды (мок-версия)
@tool
def get_weather_forecast(location: str, date: str) -> dict:
    """
    Получить прогноз погоды для указанного места и даты.
    
    Args:
        location: Город для прогноза погоды (например, "Berlin", "Moscow")
        date: Дата в формате YYYY-MM-DD
    
    Returns:
        Словарь с почасовыми температурами
    """
    # Мок-данные для демонстрации
    mock_weather = {
        "Berlin": {
            "2025-08-02T00:00": 15.5, "2025-08-02T01:00": 15.2, "2025-08-02T02:00": 14.8,
            "2025-08-02T03:00": 14.5, "2025-08-02T04:00": 14.2, "2025-08-02T05:00": 14.0,
            "2025-08-02T06:00": 14.5, "2025-08-02T07:00": 15.8, "2025-08-02T08:00": 17.2,
            "2025-08-02T09:00": 19.5, "2025-08-02T10:00": 21.8, "2025-08-02T11:00": 23.2,
            "2025-08-02T12:00": 24.5, "2025-08-02T13:00": 25.1, "2025-08-02T14:00": 25.8,
            "2025-08-02T15:00": 25.2, "2025-08-02T16:00": 24.6, "2025-08-02T17:00": 23.8,
            "2025-08-02T18:00": 22.5, "2025-08-02T19:00": 21.2, "2025-08-02T20:00": 19.8,
            "2025-08-02T21:00": 18.5, "2025-08-02T22:00": 17.2, "2025-08-02T23:00": 16.5
        },
        "Moscow": {
            "2025-08-02T00:00": 18.5, "2025-08-02T01:00": 18.0, "2025-08-02T02:00": 17.5,
            "2025-08-02T03:00": 17.0, "2025-08-02T04:00": 16.8, "2025-08-02T05:00": 16.5,
            "2025-08-02T06:00": 17.2, "2025-08-02T07:00": 18.8, "2025-08-02T08:00": 20.5,
            "2025-08-02T09:00": 22.8, "2025-08-02T10:00": 25.2, "2025-08-02T11:00": 27.5,
            "2025-08-02T12:00": 28.8, "2025-08-02T13:00": 29.5, "2025-08-02T14:00": 30.2,
            "2025-08-02T15:00": 29.8, "2025-08-02T16:00": 29.2, "2025-08-02T17:00": 28.5,
            "2025-08-02T18:00": 27.2, "2025-08-02T19:00": 25.8, "2025-08-02T20:00": 24.2,
            "2025-08-02T21:00": 22.8, "2025-08-02T22:00": 21.5, "2025-08-02T23:00": 20.2
        }
    }
    
    # Возвращаем данные для указанного города или дефолтные для Москвы
    city_data = mock_weather.get(location, mock_weather["Moscow"])
    return city_data

# Список инструментов
tools = [get_weather_forecast]
tools_by_name = {tool.name: tool for tool in tools}

# Инициализируем модель
model = ChatOpenAI(
    model="gpt-4.1",
    api_key=os.getenv("OPENAI_API_KEY"),
    temperature=0
).bind_tools(tools)

# Определяем функцию для вызова инструментов
def call_tool(state: AgentState):
    outputs = []
    # Итерируемся по вызовам инструментов в последнем сообщении
    for tool_call in state["messages"][-1].tool_calls:
        # Получаем инструмент по имени
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=str(tool_result),
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}

# Определяем функцию для вызова модели
def call_model(state: AgentState):
    # Вызываем модель с сообщениями
    response = model.invoke(state["messages"])
    # Возвращаем список, потому что он будет добавлен к существующим сообщениям
    return {"messages": [response]}

# Определяем условное ребро, которое определяет, продолжать или нет
def should_continue(state: AgentState):
    messages = state["messages"]
    # Если последнее сообщение не является вызовом инструмента, то завершаем
    if not messages[-1].tool_calls:
        return "end"
    # по умолчанию продолжаем
    return "continue"

# Создаем граф
def create_react_agent():
    """Создает и возвращает скомпилированный ReAct агент"""
    
    # Определяем новый граф с нашим состоянием
    workflow = StateGraph(AgentState)
    
    # 1. Добавляем наши узлы
    workflow.add_node("llm", call_model)
    workflow.add_node("tools", call_tool)
    
    # 2. Устанавливаем точку входа как `llm`, это первый вызываемый узел
    workflow.set_entry_point("llm")
    
    # 3. Добавляем условное ребро после вызова узла `llm`
    workflow.add_conditional_edges(
        # Ребро используется после вызова узла `llm`
        "llm",
        # Функция, которая определит, какой узел будет вызван следующим
        should_continue,
        # Карта для определения, куда идти дальше
        {
            # Если `tools`, то вызываем узел инструментов
            "continue": "tools",
            # Иначе завершаем
            "end": END,
        },
    )
    
    # 4. Добавляем обычное ребро после вызова `tools`, следующим вызывается узел `llm`
    workflow.add_edge("tools", "llm")
    
    # Компилируем граф
    return workflow.compile()

# Создаем глобальный экземпляр агента
react_agent = create_react_agent()

# Функция для запуска агента
async def run_agent(user_input: str) -> str:
    """
    Запускает ReAct агента с пользовательским вводом
    
    Args:
        user_input: Запрос пользователя
        
    Returns:
        Ответ агента
    """
    # Создаем начальное сообщение
    inputs = {"messages": [HumanMessage(content=user_input)]}
    
    # Запускаем граф
    result = react_agent.invoke(inputs)
    
    # Возвращаем последнее сообщение от ассистента
    last_message = result["messages"][-1]
    if hasattr(last_message, 'content'):
        return last_message.content
    else:
        return str(last_message) 