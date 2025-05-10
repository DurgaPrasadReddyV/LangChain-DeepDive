import os

from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain.schema import BaseMessage
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
    template = "Tell me a joke about {topic}."
    prompt_template = ChatPromptTemplate.from_template(template)
    print("-----Prompt from Template-----")
    prompt = prompt_template.invoke({"topic": "cats"})
    print(prompt)

    template_multiple = """Tell me a {adjective} story about a {animal}."""

    prompt_template_multiple = ChatPromptTemplate.from_template(template_multiple)
    prompt_multiple = prompt_template_multiple.invoke(
        {"adjective": "funny", "animal": "panda"}
    )
    print("\n----- Prompt with Multiple Placeholders -----\n")
    print(prompt_multiple)


# Traced Functions
@observe(as_type="generation")
def execute_prompt(prompt: PromptValue) -> BaseMessage:
    result = model.invoke(prompt, config={"callbacks": [langfuse_handler]})
    return result


def main():
    print_promt_output()
    template_model = "Tell me a joke about {topic}."
    prompt_template_model = ChatPromptTemplate.from_template(template_model)
    result_model = execute_prompt(prompt_template_model.invoke({"topic": "cats"}))
    print(f"Complete Response: {result_model}")
    print(f"AI: {result_model.content}")

    template_multiple_model = """Tell me a {adjective} story about a {animal}."""
    prompt_template_multiple_model = ChatPromptTemplate.from_template(
        template_multiple_model
    )
    result_multiple_model = execute_prompt(
        prompt_template_multiple_model.invoke({"adjective": "funny", "animal": "panda"})
    )
    print(f"Complete Response: {result_multiple_model}")
    print(f"AI: {result_multiple_model.content}")

    langfuse_handler.flush()


if __name__ == "__main__":
    main()
