import os
from typing import List

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langfuse.decorators import langfuse_context, observe
from pydantic import SecretStr

load_dotenv()

langfuse_context.configure(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
    enabled=True,
)

google_api_key = os.getenv("GOOGLE_API_KEY")
if not google_api_key:
    raise ValueError("GOOGLE_API_KEY environment variable is not set.")

embedding_model_name = os.getenv("EMBEDDING_MODEL")
if not embedding_model_name:
    raise ValueError("EMBEDDING_MODEL environment variable is not set.")

embedding_model = GoogleGenerativeAIEmbeddings(
    google_api_key=SecretStr(google_api_key), model=embedding_model_name
)


@observe(as_type="generation")
def generate_embeddings_documents(documents: List[str]) -> List[List[float]]:
    # automatically logged as a generation span
    langfuse_context.update_current_observation(
        input=documents, model=embedding_model.model
    )
    result = embedding_model.embed_documents(documents)
    langfuse_context.update_current_observation(output=result)
    return result


def main():
    # Invoke the model with a message
    documents = [
        "Delhi is the capital of India",
        "Kolkata is the capital of West Bengal",
        "Paris is the capital of France",
    ]
    result = generate_embeddings_documents(documents)
    print(result)
    langfuse_context.flush()


if __name__ == "__main__":
    main()
