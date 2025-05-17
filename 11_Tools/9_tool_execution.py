import os

import requests
from dotenv import load_dotenv
from langchain.schema import HumanMessage
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


@tool
def get_conversion_factor(base_currency: str, target_currency: str) -> float:
    """
    This function fetches the currency conversion factor between a given base currency and a target currency
    """
    url = f"https://v6.exchangerate-api.com/v6/c754eab14ffab33112e380ca/pair/{base_currency}/{target_currency}"

    response = requests.get(url, verify=False)

    return response.json()


@tool
def convert(base_currency_value: int, conversion_rate: float) -> float:
    """
    given a currency conversion rate this function calculates the target currency value from a given base currency value
    """

    return base_currency_value * conversion_rate


def main():
    model_with_tools = model.bind_tools([get_conversion_factor, convert])
    messages = []
    greetLLM = HumanMessage("Hi how are you")
    messages.append(greetLLM)

    greet_llm_response = model_with_tools.invoke(
        messages,
        config={"callbacks": [langfuse_handler]},
    )

    messages.append(greet_llm_response)

    messages.append(HumanMessage("What is the conversion factor from INR to USD?"))
    conversion_question_llm_response = model_with_tools.invoke(
        messages,
        config={"callbacks": [langfuse_handler]},
    )
    messages.append(conversion_question_llm_response)
    conversion_tool_message = get_conversion_factor.invoke(
        conversion_question_llm_response.tool_calls[0]
    )
    messages.append(conversion_tool_message)

    conversion_llm_response = model_with_tools.invoke(
        messages,
        config={"callbacks": [langfuse_handler]},
    )
    print(conversion_llm_response.content)
    messages.append(conversion_llm_response)
    messages.append(
        HumanMessage(
            "Now that we know the conversion rate, What is value of 1500 inr in usd?"
        )
    )
    rupee_conversion_question_llm_response = model_with_tools.invoke(
        messages,
        config={"callbacks": [langfuse_handler]},
    )
    messages.append(rupee_conversion_question_llm_response)
    rupee_conversion_tool_message = convert.invoke(
        rupee_conversion_question_llm_response.tool_calls[0]
    )
    messages.append(rupee_conversion_tool_message)
    rupee_conversion_llm_response = model_with_tools.invoke(
        messages,
        config={"callbacks": [langfuse_handler]},
    )
    print(rupee_conversion_llm_response.content)
    messages.append(rupee_conversion_llm_response)
    for message in messages:
        print(message)
    langfuse_handler.flush()


if __name__ == "__main__":
    main()
