import os

from dotenv import load_dotenv
from langchain.chat_models import init_chat_model
from langfuse.callback import CallbackHandler

load_dotenv()

langfuse_handler = CallbackHandler(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
)

model = init_chat_model(
    model_provider="google-genai",
    model=os.getenv("LLM_MODEL"),
    api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
)

prompt = "What is 30 divided by 8?"


def main():
    # Invoke the model with a message
    result = model.invoke(prompt, config={"callbacks": [langfuse_handler]})
    print("Full result:")
    print(result)
    print("Content only:")
    print(result.content)
    langfuse_handler.flush()


if __name__ == "__main__":
    main()
