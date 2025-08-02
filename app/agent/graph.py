import os
from langchain_openai import ChatOpenAI
from geopy.geocoders import Nominatim
from dotenv import load_dotenv
from langchain_core.tools import tool 
from typing import Annotated,Sequence,TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import END, StateGraph
from langgraph.graph.message import add_messages # helper function to add messages to the state
from pydantic import BaseModel, Field
import requests
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage, ToolMessage
import json
from langchain_core.runnables import RunnableConfig

geolocator = Nominatim(user_agent="weather-app")
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")


class AgentState (TypedDict):
    messages: Annotated[Sequence[BaseMessage], add_messages]
    number_of_steps: int

class SearchInput(BaseModel):
    location:str = Field(description="Город, для котрого требуется узнать прогноз погоды")
    date:str = Field(description="Дата предсказания погоды, в формате (yyyy-mm-dd). Возвращает список словарей со временем и температурой на каждый час")

#Какая погода завтра в Уваровке в 12 часов?
#llm: Уваровка, 2025-08-03
#tool:словари
#llm:дождь в 12 часов

#как устроен декоратор?
@tool("get_weather_forecast", args_schema=SearchInput, return_direct=True)
def get_weather_forecast(location:str, date:str):
    """Retrieves the weather using Open-Meteo API for a given location (city) and a date (yyyy-mm-dd). Returns a list dictionary with the time and temperature for each hour."""
    
    location = geolocator.geocode(location)
    if location:
        try:
            response = requests.get(f"https://api.open-meteo.com/v1/forecast?latitude={location.latitude}&longitude={location.longitude}&hourly=temperature_2m&start_date={date}&end_date={date}")
            data = response.json()
            return {time: temp for time, temp in zip(data["hourly"]["time"], data["hourly"]["temperature_2m"])}
        except Exception as error:
            return {"error": str(error)}
    else:
        return {"error": "Location not found"}
    



tools = [get_weather_forecast]

llm = ChatOpenAI(
    model="gpt-4.1",
    api_key=api_key
)

#print("1.")
#print(llm.invoke("Какая будет погода в Голицыно 03.08.2025?").content,"\n\n")

llm_with_tools = llm.bind_tools(tools)

SYSTEM_PROMPT = """
    Ты полезный ассистент, 
    у тебя есть инструменты для поиска информации о погоде, используй, если нужно
"""

system_message = SystemMessage(content=SYSTEM_PROMPT)
human_message = HumanMessage(content="Какая будет погода в Голицыно 03.08.2025?")

messages = [system_message, human_message]

tools_by_name = {tool.name: tool for tool in tools}


#print("2.")
#какие тулы нужно использовать и какие аргументы должны быть заполнены
#tool_call = llm_with_tools.invoke(SYSTEM_PROMPT + "Какая будет погода в Голицыно 03.08.2025?").tool_calls[0]
#даем llm выполнить функцию
# tool_result = tools[0].invoke(tool_call["args"])
# print(tool_result)

def call_tool(state: AgentState):
    outputs = []
    for tool_call in state["messages"][-1].tool_calls:
        tool_result = tools_by_name[tool_call["name"]].invoke(tool_call["args"])
        outputs.append(
            ToolMessage(
                content=tool_result,
                name=tool_call["name"],
                tool_call_id=tool_call["id"],
            )
        )
    return {"messages": outputs}

def call_model(state: AgentState, config: RunnableConfig):
    print("\n\n 123\n\n")
    response = llm.invoke(state["messages"], config)
    print(response, "\n\n 123\n\n")
    return {"messages", [response]}

def should_continue(state: AgentState):
    messages = state["messages"]
    if not messages[-1].tool_calls:
        return "end"
    return "continue"


graph = StateGraph(AgentState)

graph.add_node("call_model", call_model)
graph.add_node("call_tool", call_tool)

graph.set_entry_point("call_model")

graph.add_conditional_edges(
    "call_model",
    should_continue,
    {
        "continue": "call_tool",
        "end": END
    }
)

graph.add_edge("call_tool","call_model")

compiled_graph = graph.compile()


res = compiled_graph.invoke({"messages": [SystemMessage(content=SYSTEM_PROMPT),HumanMessage(content="Какая будет погода в Голицыно 03.08.2025?")]})

print(res)

