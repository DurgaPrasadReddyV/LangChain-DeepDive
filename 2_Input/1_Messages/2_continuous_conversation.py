import os
from typing import List

from dotenv import load_dotenv
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
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

chat_history = []  # Use a list to store messages

# Set an initial system message (optional)
system_message = SystemMessage(content="You are a helpful Calculator.")
chat_history.append(system_message)  # Add system message to chat history


# Traced Functions
@observe(as_type="generation")
def execute_messages(messages: List[str]) -> BaseMessage:
    result = model.invoke(messages, config={"callbacks": [langfuse_handler]})
    return result


def main():
    while True:
        query = input("You: ")
        if query.lower() == "exit":
            break
        chat_history.append(HumanMessage(content=query))  # Add user message

        # Get AI response using history
        result = execute_messages(chat_history)
        response = result.content
        chat_history.append(AIMessage(content=response))  # Add AI message

        print(f"AI: {response}")

    print("---- Message History ----")
    print(chat_history)

    langfuse_handler.flush()


if __name__ == "__main__":
    main()
