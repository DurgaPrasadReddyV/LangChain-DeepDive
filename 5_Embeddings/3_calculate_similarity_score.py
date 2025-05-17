import os
from typing import List

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langfuse.decorators import langfuse_context, observe
from numpy import ndarray
from pydantic import SecretStr
from sklearn.metrics.pairwise import cosine_similarity

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


# Traced Functions
@observe(as_type="generation")
def calculate_similarity(query: str, documents: List[str]) -> ndarray:
    # automatically logged as a generation span
    langfuse_context.update_current_observation(
        input={"docs": documents, "query": query}, model=embedding_model.model
    )
    doc_embeddings = embedding_model.embed_documents(documents)
    query_embedding = embedding_model.embed_query(query)
    import numpy as np

    scores = cosine_similarity(np.array([query_embedding]), np.array(doc_embeddings))

    langfuse_context.update_current_observation(
        output={"doc_embeddings": doc_embeddings, "query_embedding": query_embedding}
    )
    return scores


def main():
    documents = [
        "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
        "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
        "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
        "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
        "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers.",
    ]

    query = "tell me about bumrah"
    scores = calculate_similarity(query, documents)
    scores_with_index = list(enumerate(scores.flatten()))
    sorted_scores = sorted(scores_with_index, key=lambda x: x[1], reverse=True)
    index, score = sorted_scores[0]
    print(query)
    print(documents[index])
    print("similarity score is:", score)
    langfuse_context.flush()


if __name__ == "__main__":
    main()
