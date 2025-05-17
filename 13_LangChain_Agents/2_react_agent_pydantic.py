import ast
import os
from typing import Any

import requests
from dotenv import load_dotenv
from langchain import hub
from langchain.agents import AgentExecutor, create_react_agent
from langchain.tools import StructuredTool
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse.callback import CallbackHandler
from pydantic import BaseModel, model_validator

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


class CurrencyInput(BaseModel):
    base_currency: str
    target_currency: str

    @model_validator(mode="before")
    def parse_nested(cls, values):
        # Only run if values is a dict (as expected normally)
        if isinstance(values, dict):
            base = values.get("base_currency")
            if isinstance(base, str) and base.strip().startswith("{"):
                try:
                    parsed = ast.literal_eval(base)
                    values.update(parsed)
                except Exception as e:
                    raise ValueError("Failed to parse nested dictionary string.") from e
        return values


def get_conversion_factor(currency_input: Any) -> float:
    """
    Get the conversion rate between two currencies.

    Args:
        currency_input: Can be a CurrencyInput, dict, or string representation.
             Required keys: base_currency (e.g., 'INR') and target_currency (e.g., 'USD')

    Returns:
        float: The conversion rate
    """
    # If input is a string, try to parse it as a dict.
    if isinstance(currency_input, str):
        try:
            currency_input = CurrencyInput(**ast.literal_eval(currency_input))
        except Exception as e:
            raise ValueError("Failed to parse currency input string.") from e
    elif isinstance(currency_input, dict):
        currency_input = CurrencyInput(**currency_input)

    url = (
        f"https://v6.exchangerate-api.com/v6/c754eab14ffab33112e380ca/"
        f"pair/{currency_input.base_currency}/{currency_input.target_currency}"
    )
    response = requests.get(url, verify=False)
    data = response.json()
    return float(data["conversion_rate"])


# Set the args_schema so that inputs are parsed as CurrencyInput
get_conversion_factor_tool = StructuredTool.from_function(
    func=get_conversion_factor,
    name="get_conversion_factor",
    description="Get the currency conversion rate between two currencies. Example: {'base_currency': 'INR', 'target_currency': 'USD'}",
    args_schema=CurrencyInput,
)


class ConvertInput(BaseModel):
    base_currency_value: int
    conversion_rate: float

    @model_validator(mode="before")
    def parse_nested(cls, values):
        if isinstance(values, str):
            try:
                values = ast.literal_eval(values)
            except Exception as e:
                raise ValueError("Failed to parse convert input string.") from e
        return values


@tool
def convert(convert_input: Any) -> float:
    """
    Given a conversion input containing base_currency_value and conversion_rate,
    calculate the target currency value.
    """
    if isinstance(convert_input, str):
        try:
            convert_input = ConvertInput(**ast.literal_eval(convert_input))
        except Exception as e:
            raise ValueError("Failed to parse convert input string.") from e
    elif isinstance(convert_input, dict):
        convert_input = ConvertInput(**convert_input)
    # If already a ConvertInput, do nothing

    if isinstance(convert_input, ConvertInput):
        return float(convert_input.base_currency_value) * float(
            convert_input.conversion_rate
        )
    else:
        raise ValueError("Invalid input for conversion.")


# Step 2: Pull the ReAct prompt from LangChain Hub
prompt = hub.pull("hwchase17/react")  # pulls the standard ReAct agent prompt

# Step 3: Create the ReAct agent manually with the pulled prompt
agent = create_react_agent(
    llm=model, tools=[get_conversion_factor_tool, convert], prompt=prompt
)

# Step 4: Wrap it with AgentExecutor
agent_executor = AgentExecutor(
    agent=agent, tools=[get_conversion_factor_tool, convert], verbose=True
)


def main():
    # Step 5: Invoke
    response = agent_executor.invoke(
        {
            "input": "What is the conversion factor between INR and USD, and based on that can you convert 500 inr to usd"
        },
        config={"callbacks": [langfuse_handler]},
    )
    print(response)
    langfuse_handler.flush()


if __name__ == "__main__":
    main()
