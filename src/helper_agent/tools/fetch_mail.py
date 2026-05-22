"""
PROBLEM :: well they updated things and sample code of the email assistant in their github does not work anymore as 
its "raw langchain like", so i will have to manually unpack the tools and wrap them in @tool for this "new" version
well what do you know, that does not work either, we instead now have to wrap them in BaseTool class

PROBLEM :: even sending only 10 email worth of data to llm [llama-3.3-70b-versatile`] is too much for it
and hits the rate limit, so i will instead summarize the emails and send the summary to llm. Plus point to 
privacy as well, since it will be processed locally.
"""
from langchain_google_community import GmailToolkit
from langchain_google_community.gmail.utils import get_gmail_credentials, build_gmail_service, get_google_credentials
from crewai.tools import BaseTool
from pydantic import BaseModel, PrivateAttr
from typing import Any, Type
import ollama


class SummarizingGmailTool(BaseTool):
    name: str = ""
    description: str = ""
    args_schema: Type[BaseModel] = None
    _lc_tool: Any = PrivateAttr(default=None)

    def __init__(self, lc_tool, **kwargs):
        super().__init__(**kwargs)
        self._lc_tool = lc_tool

    def _run(self, **kwargs) -> str:
        kwargs['max_results'] = min(int(kwargs.get('max_results', 5)), 3)
        raw_result = self._lc_tool.run(kwargs)
        response = ollama.chat(
            model='llama3.2',
            messages=[{'role': 'user', 'content': 
                f"""Extract and summarize each email to maximum of 500 words each. Return a JSON array only, no other text:
                [
                    {{
                        "sender": "email address of sender",
                        "subject": "subject line",
                        "body": "concise summary of email content"
                    }}
                ]
                    Emails:
                    {raw_result}"""
                }], format='json') # forcing a json response so it can be translated to pydantic later if required, with ease
        with open(r"C:\Users\itizs\VSC\crewai\helper_agent\tests\sumarized.txt", 'w') as f:
            f.write(response['message']['content'])
        return response['message']['content']


class PassthroughGmailTool(BaseTool):
    name: str = ""
    description: str = ""
    args_schema: Type[BaseModel] = None
    _lc_tool: Any = PrivateAttr(default=None) # cuz _names fail??

    def __init__(self, lc_tool, **kwargs): 
        super().__init__(**kwargs)
        self._lc_tool = lc_tool

    def _run(self, **kwargs) -> str:
        return self._lc_tool.run(kwargs)


def get_crew_gmail_tools():
    SCOPES = ["https://www.googleapis.com/auth/gmail.readonly"]
    CLIENT_SECRET_FILE = r"client_secret_269251185509-tbecd0tdd7tbsn5l916abtcqpmpujbti.apps.googleusercontent.com.json"

    # 1. Get Credentials 
    my_credentials = get_google_credentials(token_file='token.json', client_secrets_file=CLIENT_SECRET_FILE, scopes=SCOPES)
                                                    # IF this token.json does not exist yet, it will be generated AFTER my run and login to account
    # 2. Create API
    api = build_gmail_service(credentials=my_credentials) # builds the api
    gmail_langchain_tools = GmailToolkit(api_resource=api) # takes the gmail api and wraps it to model(llm) - usable wrapper

    SUMMARIZE_TOOLS = ['search_gmail', 'get_gmail_message', 'get_gmail_thread']
    ALLOWED_TOOLS = ['search_gmail', 'get_gmail_message', 'get_gmail_thread']
    crew_ver_tools = [
        (SummarizingGmailTool if t.name in SUMMARIZE_TOOLS else PassthroughGmailTool)(
            name=t.name,
            description=t.description,
            args_schema=t.args_schema,
            lc_tool=t
        )
        for t in gmail_langchain_tools.get_tools() if t.name in ALLOWED_TOOLS 
        # while api generated on google cloud allows read only access, let us solidify it by giving it only read api
        # may add other functionality in future using PassthroughGmailTool
    ]
    return crew_ver_tools