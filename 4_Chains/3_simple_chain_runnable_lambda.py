import os

from dotenv import load_dotenv
from langchain.schema.runnable import RunnableLambda, RunnableSequence
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
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

parser = StrOutputParser()

# Define prompt templates (no need for separate Runnable chains)
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a comedian who tells jokes about {topic}."),
        ("human", "Tell me {joke_count} jokes."),
    ]
)


def main():
    format_prompt = RunnableLambda(lambda x: prompt_template.format_prompt(**x))
    invoke_model = RunnableLambda(lambda x: model.invoke(x.to_messages()))
    parse_output = RunnableLambda(lambda x: x.content)
    chain = RunnableSequence(
        first=format_prompt, middle=[invoke_model], last=parse_output
    )
    result = chain.invoke(
        {"topic": "lawyers", "joke_count": 3}, config={"callbacks": [langfuse_handler]}
    )
    print(result)
    langfuse_handler.flush()


if __name__ == "__main__":
    main()
