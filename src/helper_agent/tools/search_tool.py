from crewai.tools import BaseTool
from langchain_community.tools.yahoo_finance_news import YahooFinanceNewsTool
from langchain_community.tools import DuckDuckGoSearchResults

class yf_search(BaseTool):
    """Native CrewAI wrapper for LangChain's Yahoo Finance tool."""
    name: str = "Yahoo Finance News Search" # i can't do name = YahooFinanceNewsTool.name
    description: str = "Useful for finding recent news and financial articles using a stock ticker." # i can't do name = YahooFinanceNewsTool.description
    # the reason being that pydantic V2 was rewritten completely in rust from scratch. And since
    # our class inheris from BaseTool, pydantic takes control on how to build this class.
    # it then tries to build a RUST validation tree, while doing so, it encounters "name = OLD.name"
    # and sees pydantic V1 as completely alien object which doesn't match its V2 cux V1 returns a 
    # field Object instead of just handing it string, so it says, wtf is this, fuck you, should've 
    # studied rust instead of C++

    # Notice we didn't provide args_schema, even though they said in sample code to give one?
    # turns out they make one themselves(crewAI) How???
    # it looks at our _run() arguments, and sees a string
    # even sample code takes a string in _run(). 
    # if we do _run(*args **kwargs) it again tries to look at run and say "wtf did he pass in here"
    # and if we try to specify using CLASS.args_schema, it fails cuz of pydantic. So accept string supremacy
    def _run(self, query: str) -> str:
        # 1. Initialize the tool cleanly don't worry about pydantic here, it has no power in here MUHAHAHAHAH (it doesn't check functions)
        # we actually have to copy and recreate the whole schema in pydantic V2 to pass here, fortunately this one accepts only one - query
        langchain_tool = YahooFinanceNewsTool()
        # 2. Pass the agent's query into the execution method
        return langchain_tool.invoke(query)
    
class ddg_search(BaseTool):
    """Native CrewAI wrapper for LangChain's DuckDuckGo tool."""
    name: str = "DuckDuckGo Internet Search"
    description: str = "A wrapper around DuckDuckGo Search. Useful for general internet research."

    def _run(self, query: str) -> str:
        langchain_tool = DuckDuckGoSearchResults()
        return langchain_tool.invoke(query)