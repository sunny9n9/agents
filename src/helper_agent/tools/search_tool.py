from os import getenv
import requests
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
    
class jina_search(BaseTool):
    name : str = "Scrape Websites With Jina Links"
    description : str = "To get website content cleanly without header footer ads etc."
    api : str = getenv("JINA_API_KEY")
    
    def _run(self, query: str) -> str:
        try:
            if not query.startswith("http"):
                query = f"https://{query}"
            
            jina_url = f"https://r.jina.ai/{query}"
                
            # jina filters
            headers = {
                "Authorization" : f"Bearer {self.api}",
                "X-With-Generated-Alt": "true",
                "X-With-Links-Summary": "false",
                "X-Return-Format": "text",
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
                "X-Remove-Selector": (
                    "header, footer, nav, aside, .ads, .advertisement, .banner, "
                    ".sidebar, .popup, .modal, .newsletter, .subscribe, "
                    ".comments, .social-share, .related, .recommended, "
                    ".cookie, .cookie-banner, .paywall, script, style, noscript"
                ),
                "X-Target-Selector": "article, main",
            }
            response = requests.get(
                url=jina_url,
                headers=headers,
                timeout=20
            )

            response.raise_for_status() # convert http failures into python exception
            return response.text[:50000] # a very good idea - cap the response to save tokens ##################

        except requests.exceptions.RequestException as e:
            return f"[ERROR] :: Jina fetch failed :: {e}"