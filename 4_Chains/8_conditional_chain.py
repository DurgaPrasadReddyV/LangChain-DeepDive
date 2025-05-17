import os

from dotenv import load_dotenv
from langchain.schema.runnable import RunnableBranch
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

# Define prompt templates for different feedback types
positive_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Generate a thank you note for this positive feedback: {feedback}."),
    ]
)

negative_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        ("human", "Generate a response addressing this negative feedback: {feedback}."),
    ]
)

neutral_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        (
            "human",
            "Generate a request for more details for this neutral feedback: {feedback}.",
        ),
    ]
)

escalate_feedback_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        (
            "human",
            "Generate a message to escalate this feedback to a human agent: {feedback}.",
        ),
    ]
)

# Define the feedback classification template
classification_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant."),
        (
            "human",
            "Classify the sentiment of this feedback as positive, negative, neutral, or escalate: {feedback}. Provide feedback in small case.",
        ),
    ]
)


def main():
    # Define the runnable branches for handling feedback. Looks like branching is case sensitive. mention case in classification prompt explicitly
    branches = RunnableBranch(
        (
            lambda x: "positive" in x,
            positive_feedback_template | model | parser,  # Positive feedback chain
        ),
        (
            lambda x: "negative" in x,
            negative_feedback_template | model | parser,  # Negative feedback chain
        ),
        (
            lambda x: "neutral" in x,
            neutral_feedback_template | model | parser,  # Neutral feedback chain
        ),
        escalate_feedback_template | model | parser,
    )

    # Create the classification chain
    classification_chain = classification_template | model | parser

    # Combine classification and response generation into one chain
    chain = classification_chain | branches

    # Run the chain with an example review
    # Good review - "The product is excellent. I really enjoyed using it and found it very helpful."
    # Bad review - "The product is terrible. It broke after just one use and the quality is very poor."
    # Neutral review - "The product is okay. It works as expected but nothing exceptional."
    # Default - "I'm not sure about the product yet. Can you tell me more about its features and benefits?"

    review = "The product is terrible. It broke after just one use and the quality is very poor."
    result = chain.invoke(
        {"feedback": review}, config={"callbacks": [langfuse_handler]}
    )
    chain.get_graph().print_ascii()
    print(result)
    langfuse_handler.flush()


if __name__ == "__main__":
    main()
