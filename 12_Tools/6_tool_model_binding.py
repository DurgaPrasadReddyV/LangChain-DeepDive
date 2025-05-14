import os

from dotenv import load_dotenv
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
    result = model_with_tools.invoke(
        "Hi how are you",
        config={"callbacks": [langfuse_handler]},
    )
    print(result)
    print(multiply_tool.name)
    print(multiply_tool.description)
    print(multiply_tool.args)
    print(multiply_tool.args_schema.model_json_schema())
    langfuse_handler.flush()


if __name__ == "__main__":
    main()
