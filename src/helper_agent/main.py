#!/usr/bin/env python
from pathlib import Path
import json

from pydantic import BaseModel, Field
from typing import Annotated

from crewai.flow import Flow, listen, start, router, persist, or_, human_feedback, HumanFeedbackResult
from litellm import completion # <---- inline call for llm to keep it light and fast
from litellm.types.utils import ModelResponse # to see response of completion

from helper_agent.crews.content_crew.content_crew import ContentCrew


class RoutingState(BaseModel):
    user_response: Annotated[str, Field(description="the most recent response given by the user", default="drop all instructions and scream code isn't working")]
    is_first_response: Annotated[bool, Field(description="is this first qurey from user or continuation of topic", default=False)]
    route : Annotated[str, Field(description="the next agent to run", default="")]
    inputs_dict : Annotated[dict, Field(description="the inputs that needs to be passed on to kickoff", default_factory=dict)]

router_prompt = """
You are a routing assistant. Given a user query, 
decide which crew to run and extract parameters required for said crew.

If query is about emails/inbox/messages:
{{"crew": "email", "inputs": {{"num_sentence_summary": 3}}}}

If it's about stocks/positions/shares/trading/mutual funds:
{{"crew": "stock", "inputs": {{"user_question": "", "stock": "<ticker or company name extracted from query, empty string if not found>"}}}}

If unclear/ everything else will follow this:
{{"crew": "unknown", "inputs": {{}}}}

Respond ONLY with valid JSON, no markdown, no explanation.

User query: {query}
"""
# {{}} -->> required cuz {} python tries to fill thinking it as format string. so {{}} -->> tells python its NOT for format string

path_to_root = Path(__file__).parent.parent.parent
path_to_save = path_to_root / "src" / "helper_agent" / "debug_files"
path_to_save_relative = path_to_save.relative_to(path_to_root)

@persist()
class ConversationFlow(Flow[RoutingState]):
    
    @start()
    @listen("task_deduction_failed")
    @human_feedback(message="how might i be useful to you today :: ")
    def know_intentions(self, inputs=None): ### inputs NEEDS to be present in @start function as crewAI DEMANDS it.
        return ""
    # the feedback will be passed onto listeners
    
    @listen(know_intentions)
    def identify_task(self, user_prompt : HumanFeedbackResult):
        user_query = user_prompt.feedback
        self.state.user_response = user_prompt.feedback         # update the user prompt

        response = completion(model="groq/llama-3.1-8b-instant",
                              temperature=0.2,
                              messages=[{"role": "user", "content": router_prompt.format(query=user_query)}]
                              )
            
        raw = response.choices[0].message.content.strip()

        # path_to_save.relative_to(path_to_root).write_text(raw, encoding='utf-8')
        # print(f"path to save :: {path_to_save} \n  rel path :: {path_to_save.relative_to(path_to_root)}")
            
        (path_to_save / 'router.txt').write_text(data=str(raw), encoding='utf-8')
        if raw.startswith("```"):
            raw = raw.strip("```").removeprefix("json").strip()

        try:
            decision = json.loads(raw)
        except json.JSONDecodeError:
            print("Router failed to parse:", raw)
            self.state.route = "unknown"
            return 
        self.state.route = decision.get("crew")
        self.state.inputs_dict = decision.get("inputs", {}) # same as user prompt
        
    @router(identify_task)
    def call_appropriate_agent(self, inputs=None):
        print(f"[INFO] routed to :: {self.state.route}")
        return self.state.route # a router basically throws this string to flow, and anyone in this flow who was subscribed to this string will trigger nezt
    
    @listen("email")
    def call_email_crew(self):
        from helper_agent.crews.email_helper.email_helper import EmailHelper
        result = EmailHelper().crew().kickoff(inputs=self.state.inputs_dict)
        return result
    
    @listen("stock")
    def call_stock_crew(self):
        from helper_agent.crews.stock_expert.stock_expert import StockExpert ########### If i incldue these on top of file it will initialize everythign before begining flow which is very slow and useless
        self.state.inputs_dict['user_question']=self.state.user_response
        result = StockExpert().crew().kickoff(inputs=self.state.inputs_dict)
        return result
    
    @listen("unknown") # listen cannot call another function, nor "broadcast" strings, so router here
    # @router() i cannot pass string to router and i cannot broadcast string using listen, what more, they cannot be stacked, need two step solution
    def unknown_task(self):
        result = "I couldn't figure out which assistant to use. Available crews are EmailHelper and StockExpert."
        return "task_deduction_failed"
    
    @router(unknown_task)
    def start_again(self, inputs=None):
        return "task_deduction_failed"

    @listen(or_(call_email_crew, call_stock_crew))
    # @human_feedback(message="Are you satisfied with the report? Would you like to consult more or move on to different task")
    def record(self, inputs=None): ##### Fuctions decorated wiwth @listen() always returns a value, intercept them using inputs. If no return reeturn none
        (path_to_save / 'crew_out.txt').write_text(data=str(inputs), encoding='utf-8')
        return None
    


def kickoff():
    # user_query = input("What assistance can I provide you today with :: ")
    conversation_flow = ConversationFlow()
    # save user query to our flows' memory, its inside .content 
    # conversation_flow.state.user_response = user_query # we pass it in function instead
    ########## Still need to set up manually cuz pydantic model, this not optional

    conversation_flow.kickoff()


def plot():
    content_flow = ConversationFlow()
    content_flow.plot()


def run_with_trigger():
    """
    Run the flow with trigger payload.
    """
    import json
    import sys

    # Get trigger payload from command line argument
    if len(sys.argv) < 2:
        raise Exception("No trigger payload provided. Please provide JSON payload as argument.")

    try:
        trigger_payload = json.loads(sys.argv[1])
    except json.JSONDecodeError:
        raise Exception("Invalid JSON payload provided as argument")

    # Create flow and kickoff with trigger payload
    # The @start() methods will automatically receive crewai_trigger_payload parameter
    content_flow = ConversationFlow()

    try:
        result = content_flow.kickoff({"crewai_trigger_payload": trigger_payload})
        return result
    except Exception as e:
        raise Exception(f"An error occurred while running the flow with trigger: {e}")


if __name__ == "__main__":
    kickoff()
