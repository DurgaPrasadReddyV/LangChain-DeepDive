import os

from dotenv import load_dotenv
from langchain_core.prompts import PromptTemplate
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

# 1st prompt -> detailed report
report_template = PromptTemplate(
    template="Write a detailed report on {topic}", input_variables=["topic"]
)

# 2nd prompt -> summary
summary_template = PromptTemplate(
    template="Write a 5 line summary on the following text. /n {text}",
    input_variables=["text"],
)


def main():
    report_prompt = report_template.invoke({"topic": "black hole"})

    report_result = model.invoke(
        report_prompt, config={"callbacks": [langfuse_handler]}
    )

    summary_prompt = summary_template.invoke({"text": report_result.content})

    summary_result = model.invoke(
        summary_prompt, config={"callbacks": [langfuse_handler]}
    )

    print(summary_result.content)
    langfuse_handler.flush()


if __name__ == "__main__":
    main()
