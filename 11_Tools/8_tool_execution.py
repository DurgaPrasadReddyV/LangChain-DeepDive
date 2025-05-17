import os

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
def multiply_tool(a: int, b: int) -> int:
    """Given 2 numbers a and b this tool returns their product"""
    return a * b


def main():
    model_with_tools = model.bind_tools([multiply_tool])
    greetLLM = HumanMessage("Hi how are you")
    messages = [greetLLM]

    greet_llm_response = model_with_tools.invoke(
        messages,
        config={"callbacks": [langfuse_handler]},
    )

    messages.append(greet_llm_response)
    messages.append(HumanMessage("Can you multiply 4 and 5?"))
    multiply_llm_response = model_with_tools.invoke(
        messages,
        config={"callbacks": [langfuse_handler]},
    )
    messages.append(multiply_llm_response)
    multiply_tool_result = multiply_tool.invoke(multiply_llm_response.tool_calls[0])
    messages.append(multiply_tool_result)
    multiply_llm_result = model_with_tools.invoke(
        messages,
        config={"callbacks": [langfuse_handler]},
    )
    print(multiply_llm_result.content)
    messages.append(multiply_llm_result)

    messages.append(HumanMessage("Can you divide 4 and 5?"))
    divide_llm_response = model_with_tools.invoke(
        messages,
        config={"callbacks": [langfuse_handler]},
    )
    messages.append(divide_llm_response)

    messages.append(
        HumanMessage(
            "Can you try using your knowledge about division instead of relying on tools and provide an answer?"
        )
    )
    divide_llm_response2 = model_with_tools.invoke(
        messages,
        config={"callbacks": [langfuse_handler]},
    )

    messages.append(divide_llm_response2)

    divide_llm_without_tools_response = model.invoke(
        [HumanMessage("Can you divide 4 and 5?")],
        config={"callbacks": [langfuse_handler]},
    )
    messages.append(divide_llm_without_tools_response)

    for message in messages:
        print(message)
    langfuse_handler.flush()


if __name__ == "__main__":
    main()
