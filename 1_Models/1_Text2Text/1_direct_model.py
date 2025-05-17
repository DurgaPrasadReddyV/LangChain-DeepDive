import os

from dotenv import load_dotenv
from google import genai
from langfuse.decorators import langfuse_context, observe

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

langfuse_context.configure(
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    host=os.getenv("LANGFUSE_HOST"),
    enabled=True,
)

prompt = "What is 30 divided by 8?"


@observe(as_type="generation")
def execute_prompt(prompt: str, model: str) -> str:
    langfuse_context.update_current_observation(input=prompt, model=model)
    resp = client.models.generate_content(model=model, contents=prompt)
    langfuse_context.update_current_observation(output=resp.text)
    return resp.text if resp.text is not None else ""


def main():
    model = os.getenv("LLM_MODEL")
    if model is None:
        raise ValueError("LLM_MODEL environment variable is not set.")
    output = execute_prompt(prompt, model)
    print(output)
    langfuse_context.flush()


if __name__ == "__main__":
    main()
