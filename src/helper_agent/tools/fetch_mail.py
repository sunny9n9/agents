"""
PROBLEM :: well they updated things and sample code of the email assistant in their github does not work anymore as 
its "raw langchain like", so i will have to manually unpack the tools and wrap them in @tool for this "new" version
well what do you know, that does not work either, we instead now have to wrap them in BaseTool class

PROBLEM :: even sending only 10 email worth of data to llm [llama-3.3-70b-versatile`] is too much for it
and hits the rate limit, so i will instead summarize the emails and send the summary to llm. Plus point to 
privacy as well, since it will be processed locally.

PROBLEM :: too slow and unreliable outputs, privacy will be addressed later, back to crew(s)
"""
from langchain_google_community import GmailToolkit
from langchain_google_community.gmail.utils import get_gmail_credentials, build_gmail_service, get_google_credentials
from crewai.tools import BaseTool
from pydantic import BaseModel, PrivateAttr
from typing import Any, Type
import ollama
import json

import re # because bad format never leaves me. too many shit format in emails as well
def clean_text(text: str) -> str:
    text = re.sub(r'[\u0000-\u001f\u007f-\u009f\u00ad\u034f\u200b-\u200f\u2028\u2029\u202a-\u202e\u2060-\u206f\ufeff\ufff0-\uffff]', ' ', text)
    text = re.sub(r'https?://\S+', '[link]', text)
    text = re.sub(r'\s{2,}', ' ', text)
    return text.strip()

class SummarizingGmailTool(BaseTool):
    name: str = ""
    description: str = ""
    args_schema: Type[BaseModel] = None
    _lc_tool: Any = PrivateAttr(default=None) #### We need to specify PrivateAttr as pydantic treats _xyz as class attr, and ignores it

    def __init__(self, lc_tool, **kwargs):
        super().__init__(**kwargs)
        self._lc_tool = lc_tool

    def _run(self, **kwargs) -> str:
        kwargs['max_results'] = min(int(kwargs.get('max_results', 3)), 3) # might keep this 3 for testing
        kwargs['resource'] = 'messages'  # agent keeps choosing threads, i need all email summary not thread
        raw_result = self._lc_tool.run(kwargs) ##### Langchain expects dict passed positionally, do not use **kwargs
        
        if isinstance(raw_result, list): # LangChain returns str / list / dict / custom obj, its very changing(changes every update). AI is moving FAST
            raw_result = json.dumps(raw_result)
        elif not isinstance(raw_result, str):
            raw_result = str(raw_result)
            # next function, clean_text need STRICTLY string input, that is why this part

        raw_result = clean_text(raw_result) # remove *SOME* garbage
        ########## ------------- Trusting Another Online LLM to do this ------------------- ############
        # response = ollama.chat(
        #     model='llama3.2',
        #     messages=[{'role': 'user', 'content': 
        #         f"""You are an email parser. Extract and summarize each email from the raw data below.
        #             For each email return sender email, subject, and a few sentence body summary.
        #             Return ONLY a JSON array, no explanation, no markdown:
        #             [
        #                 {{
        #                     "sender": "sender@email.com",
        #                     "subject": "subject line",
        #                     "body": "Intents behind the email summarized to a maximum of 5 sentences"
        #                 }}
        #             ]
        #             Emails:
        #             {raw_result}"""
        #         }], format='json',    # forcing a json response so it can be translated to pydantic later if required, with ease
        #             options={'num_ctx': 8192, 'temperature': 0.0} # default is 4096, can double again, but dam i wish for graphics card
        #         )
        
        # with open(r"C:\Users\itizs\VSC\crewai\helper_agent\tests\sumarized.txt", 'w') as f:
        #     f.write(response['message']['content'])
        with open(r"C:\Users\itizs\VSC\crewai\helper_agent\src\helper_agent\debug_files\raw_mail.txt", 'w') as f:
            f.write(raw_result)
            
        # try:
        #     parsed = json.loads(response['message']['content'])
        #     if not isinstance(parsed, list):
        #         parsed = parsed.get('emails', list(parsed.values())[0] if parsed else [])
        #     return json.dumps(parsed)
        # except (json.JSONDecodeError, Exception):
        #     return '[]'
        return raw_result


class PassthroughGmailTool(BaseTool):
    name: str = ""
    description: str = ""
    args_schema: Type[BaseModel] = None
    _lc_tool: Any = PrivateAttr(default=None) # cuz _names fail??

    def __init__(self, lc_tool, **kwargs): 
        super().__init__(**kwargs)
        self._lc_tool = lc_tool

    def _run(self, **kwargs) -> str:
        return self._lc_tool.run(kwargs) ##### Langchain expects dict passed positionally, do not use **kwargs


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
    ALLOWED_TOOLS = ['search_gmail' 
                     # 'get_gmail_message', # <--- No use having this as search_gmail returns body as well
                     # 'get_gmail_thread' # <--- Not useful for my use case
                     ]
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