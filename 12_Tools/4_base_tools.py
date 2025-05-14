from typing import Type

from langchain.tools import BaseTool
from pydantic import BaseModel, Field

# arg schema using pydantic


class MultiplyInput(BaseModel):
    a: int = Field(required=True, description="The first number to add")
    b: int = Field(required=True, description="The second number to add")


class MultiplyTool(BaseTool):
    name: str = "multiply"
    description: str = "Multiply two numbers"

    args_schema: Type[BaseModel] = MultiplyInput

    def _run(self, a: int, b: int) -> int:
        return a * b


multiply_tool = MultiplyTool()


def main():
    result = multiply_tool.invoke({"a": 3, "b": 5})

    print(result)
    print(multiply_tool.name)
    print(multiply_tool.description)
    print(multiply_tool.args)
    print(multiply_tool.args_schema.model_json_schema())


if __name__ == "__main__":
    main()
