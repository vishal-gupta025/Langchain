from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.tools import ShellTool

search_tool = DuckDuckGoSearchRun()

results = search_tool.invoke("ind vs pak cricket match in asia cup 2025")

print(f"Search results:\n{results}")


# shell_tool = ShellTool()

# result = shell_tool.invoke("ls")

# print(f"Shell command result:\n{result}")