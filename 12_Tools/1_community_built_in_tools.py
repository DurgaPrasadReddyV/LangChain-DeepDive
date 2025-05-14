from dotenv import load_dotenv
from langchain_community.tools import DuckDuckGoSearchRun, ShellTool

load_dotenv()

search_tool = DuckDuckGoSearchRun()
shell_tool = ShellTool()


def main():
    search_results = search_tool.invoke("top news in india today")
    print(search_results)
    print(search_tool.name)
    print(search_tool.description)
    print(search_tool.args)

    shell_results = shell_tool.invoke("dir")  # "dir" command is specific to windows os

    print(shell_results)


if __name__ == "__main__":
    main()
