import os
from typing import List

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

# Langfuse imports
from langfuse.decorators import langfuse_context, observe

load_dotenv()

langfuse_context.configure(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
    enabled=True,
)

embedding_model = GoogleGenerativeAIEmbeddings(
    google_api_key=os.getenv("GOOGLE_API_KEY"), model=os.getenv("EMBEDDING_MODEL")
)


# Traced Functions
@observe(as_type="generation")
def generate_embeddings_documents(documents: List[str]) -> List[float]:
    # automatically logged as a generation span
    langfuse_context.update_current_observation(
        input=documents, model=embedding_model.model
    )
    result = embedding_model.embed_documents(documents)
    langfuse_context.update_current_observation(output=result)
    langfuse_context.flush()
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


if __name__ == "__main__":
    main()
