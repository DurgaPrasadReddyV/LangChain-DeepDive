from dotenv import load_dotenv
from langchain_core.tools import tool

load_dotenv()


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


def main():
    result = multiply.invoke({"a": 3, "b": 5})

    print(result)
    print(multiply.name)
    print(multiply.description)
    print(multiply.args)
    print(multiply.args_schema.model_json_schema())


if __name__ == "__main__":
    main()
