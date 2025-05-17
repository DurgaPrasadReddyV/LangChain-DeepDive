import os
from typing import List

from dotenv import load_dotenv
from langchain.schema import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langfuse.callback import CallbackHandler
from langfuse.decorators import observe

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

chat_history = []  # Use a list to store messages

# Set an initial system message (optional)
system_message = SystemMessage(content="You are a helpful Calculator.")
chat_history.append(system_message)  # Add system message to chat history

human_message = HumanMessage(content="What is 30 divided by 8?")
chat_history.append(human_message)  # Add human message to chat history


# Traced Functions
@observe(as_type="generation")
def execute_messages(messages: List[str]) -> BaseMessage:
    result = model.invoke(messages, config={"callbacks": [langfuse_handler]})
    return result


def main():
    # Invoke the model with a message
    result = execute_messages(chat_history)
    print("Full result:")
    print(result)
    print("Content only:")
    print(result.content)
    ai_message = AIMessage(content=result.content)
    chat_history.append(ai_message)  # Add ai message to chat history
    human_message2 = HumanMessage(content="What is 11 multiplied by 8?")
    chat_history.append(human_message2)  # Add 2nd human message to chat history
    result = execute_messages(chat_history)
    print("Full result:")
    print(result)
    print("Content only:")
    print(result.content)
    langfuse_handler.flush()


if __name__ == "__main__":
    main()
