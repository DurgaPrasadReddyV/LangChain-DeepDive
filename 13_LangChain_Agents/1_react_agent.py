import os

import requests
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain_community.tools import TavilySearchResults
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse.callback import CallbackHandler

load_dotenv()

langfuse_handler = CallbackHandler(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

model = ChatGoogleGenerativeAI(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    model=os.getenv("LLM_MODEL"),
    temperature=0,
)


search_tool = TavilySearchResults()


@tool
def get_weather_data(city: str) -> str:
    """
    This function fetches the current weather data for a given city
    """
    url = f"https://api.weatherstack.com/current?access_key=4d1d8ae207a8c845a52df8a67bf3623e&query={city}"

    response = requests.get(url, verify=False)

    return response.json()


# Step 2: Pull the ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")  # pulls the standard ReAct agent prompt

# Step 3: Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(
    llm=model, tools=[search_tool, get_weather_data], prompt=prompt
)

# Step 4: Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent, tools=[search_tool, get_weather_data], verbose=True
)


def main():
    # Step 5: Invoke
    response = agent_executor.invoke(
        {
            "input": "Find the capital of Madhya Pradesh, then find it's current weather condition"
        },
        config={"callbacks": [langfuse_handler]},
    )
    print(response)
    langfuse_handler.flush()


if __name__ == "__main__":
    main()
