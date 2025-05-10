import os

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage, HumanMessage
from langchain_core.prompt_values import PromptValue
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse import Langfuse
from langfuse.callback import CallbackHandler
from langfuse.decorators import observe

load_dotenv()

lf = Langfuse(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

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


def print_promt_output():
    messages = [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]

    prompt_template = ChatPromptTemplate.from_messages(messages)
    prompt = prompt_template.invoke({"topic": "lawyers", "joke_count": 3})
    print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
    print(prompt)

    messages_concrete_type = [
        ("system", "You are a comedian who tells jokes about {topic}."),
        HumanMessage(
            content="Tell me {joke_count} jokes."
        ),  # Prompt templates cannot be used with concrete messages joke_count is not resolved.
    ]

    prompt_concrete_template = ChatPromptTemplate.from_messages(messages_concrete_type)
    prompt_concrete = prompt_concrete_template.invoke(
        {"topic": "lawyers", "joke_count": 3}
    )
    print("\n----- Prompt with System and Human Messages (Tuple) -----\n")
    print(prompt_concrete)


# Traced Functions
@observe(as_type="generation")
def execute_prompt(prompt: PromptValue) -> BaseMessage:
    result = model.invoke(prompt, config={"callbacks": [langfuse_handler]})
    return result


def main():
    print_promt_output()
    messages_model = [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
    prompt_template_model = ChatPromptTemplate.from_messages(messages_model)
    result_model = execute_prompt(
        prompt_template_model.invoke({"topic": "lawyers", "joke_count": 3})
    )
    print(f"Complete Response: {result_model}")
    print(f"AI: {result_model.content}")

    langfuse_handler.flush()


if __name__ == "__main__":
    main()
