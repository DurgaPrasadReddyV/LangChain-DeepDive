from langchain_core.tools import tool


# Custom tools
@tool
def add(a: int, b: int) -> int:
    """Add two numbers"""
    return a + b


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two numbers"""
    return a * b


class MathToolkit:
    def get_tools(self):
        return [add, multiply]


def main():
    toolkit = MathToolkit()
    tools = toolkit.get_tools()

    for added_tool in tools:
        print(added_tool.name, "=>", added_tool.description)


if __name__ == "__main__":
    main()
