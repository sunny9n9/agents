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
from typing import Any, Type, DefaultDict
import ollama
import json
import pathlib

import re # because bad format never leaves me. too many shit format in emails as well
def clean_text(text: str) -> str:
    text = re.sub(r'\\u[0-9a-fA-F]{4}', ' ', text)
    text = re.sub(r'[\u0000-\u001f\u007f-\u009f\u00ad\u034f\u200b-\u200f\u2028\u2029\u202a-\u202e\u2060-\u206f\ufeff\ufff0-\uffff]', ' ', text)
    text = re.sub(r'https?://\S+', ' ', text)
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
    CLIENT_SECRET_FILE = r"client_secret_269251185509-8rqmhjv10c7la0u0esn2imfc6uhov88g.apps.googleusercontent.com.json"

    # 0. Check if creadentials authentication already exist
    TOKENS = ["token_primary", "token_secondary", "token_college"] # token (generated after auth) for my 3 mails
    account_infos = {}

    for _token in TOKENS:
        token_path = f"{_token}.json"
        my_credentials = get_google_credentials(
            token_file=token_path,  # IF this token.json does not exist yet, it will be generated AFTER my run and login to account
            client_secrets_file=CLIENT_SECRET_FILE, 
            scopes=SCOPES
        )
        
        api = build_gmail_service(credentials=my_credentials)  # builds the api
        account_infos[_token] = GmailToolkit(api_resource=api) # takes the gmail api and wraps it to model(llm) - usable wrapper
        # dictionary maps correct account to its corrosponding list of tokens

    SUMMARIZE_TOOLS = ['search_gmail', 'get_gmail_message', 'get_gmail_thread']
    ALLOWED_TOOLS = ['search_gmail' 
                     # 'get_gmail_message', # <--- No use having this as search_gmail returns body as well
                     # 'get_gmail_thread' # <--- Not useful for my use case
                     ]
    
    all_tools_flattened = [] # cuz the calling function expects a tool list given to it.
    
    # for accounts that need auth
    for account in TOKENS:
        
        # create a clean identifier suffix (e.g., "personal", "misc", "college")
        suffix = account.replace("token_", "") # change of name important cuz otherwise there will be conflict as same tools multiple copies for multiple accounts
        
        # pull the correct toolkit for this specific account
        current_toolkit = account_infos[account]
        
        for _t in current_toolkit.get_tools():
            if _t.name in ALLOWED_TOOLS:
                # to edit function/tool names and their docstring/instructions for the LLM
                unique_name = f"{_t.name}_{suffix}"
                unique_desc = f"{_t.description} ONLY searches the {suffix} email account inbox."
                
                if _t.name in SUMMARIZE_TOOLS:
                    all_tools_flattened.append(SummarizingGmailTool(
                        name=unique_name,
                        description=unique_desc,
                        args_schema=_t.args_schema,
                        lc_tool=_t
                    ))
                else:
                    all_tools_flattened.append(PassthroughGmailTool(
                        name=unique_name,
                        description=unique_desc,
                        args_schema=_t.args_schema,
                        lc_tool=_t
                    ))
        # while api generated on google cloud allows read only access, let us solidify it by giving it only read api
        # may add other functionality in future using PassthroughGmailTool such as writing draft mails.
    
    return all_tools_flattened