import os

from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
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


# Traced Functions
@observe(as_type="generation")
def execute_prompt(prompt: PromptValue) -> BaseMessage:
    result = model.invoke(prompt, config={"callbacks": [langfuse_handler]})
    return result


def main():
    # detailed way
    prompt_template = PromptTemplate(
        template="Greet this person in 5 languages. The name of the person is {name}",
        input_variables=["name"],  # usually not required. It is inferred from template.
    )

    # fill the values of the placeholders
    result = execute_prompt(prompt=prompt_template.invoke({"name": "Prasad"}))

    print(result.content)

    langfuse_handler.flush()


if __name__ == "__main__":
    main()
