from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.tasks.task_output import TaskOutput
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
from crewai.knowledge.knowledge_config import KnowledgeConfig
from ...tools.search_tool import yf_search, ddg_search
from crewai_tools import ScrapeWebsiteTool
from pathlib import Path
from time import sleep
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators
# ------------------------------------------------------------------------ [Callback(s)]
def token_limit_saver(output : TaskOutput):
    ''' taking in a TaskOutput is compulsory as crewAI will throw it towards us regardless'''
    print(f"Entring sleep of 45 sec to prevent token limit ")
    sleep(45) # there is 1 min limit i will assume prev model already took 15 sec to process
    print(f"sleep completed, continuing execution")
    ### Place this OUTSIDE class else pydantic complains checkpoint failures

@CrewBase
class StockExpert():
    """StockExpert crew"""

    agents: list[BaseAgent]
    tasks: list[Task]
    # -------------------------------------------------------------------- [Getting Path]
    CURRENT_DIR = Path(__file__).resolve().parent
    PROJECT_ROOT = CURRENT_DIR.parent.parent.parent.parent
    knowledge_base = PROJECT_ROOT / "knowledge"

    # build knowledge base
    knowledge_books = knowledge_base / "Books" # <------ more text-y pdf(s)
    # knowledge_books=pathlib.Path("knowledge/WealthCreationMO") 
    # # ------- very presentation like.mi8 need different parser
    # knowledge_config = KnowledgeConfig()
    books_list = []
    for p in knowledge_books.rglob('*.pdf'):
        books_list.append(str(p.relative_to(knowledge_base)))
        print(books_list)

    investment_books = PDFKnowledgeSource(
        chunk_size=1000,
        chunk_overlap=200,
        file_paths=books_list
    )

    # -------------------------------------------------------------------- [Getting Tool(s)]
    duck_search = ddg_search()
    yfin_search = yf_search()
    website_scrpe = ScrapeWebsiteTool() # only returns non js stuff without links T-T
    # -------------------------------------------------------------------- [Agent Definitions]
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools

    gemini = LLM(
        model="gemini/gemini-3.5-flash", # <----- lord and saviour big context winodow
        temperature=0.4,
        reasoning_effort='medium', 
    )
    gemini_lite = LLM(
        model="gemini/gemini-2.5-flash-lite",
        temperature=0.4
    )
    mixtral = LLM(
        model="groq/mixtral-8x7b-32768", ########### DEPRECIATED
        temperature=0.2 # <---- need no creativity for searching net
    )
    llama_big_old = LLM(
        model="groq/llama-3.1-70b-versatile",
        temperature=0.3 # <---- little older version, thought to trick groq token limit but doesn't work that way
    )
    llama_big_new = LLM(
        model="groq/llama-3.3-70b-versatile",
        temperature=0.3 # <---- copy of above cuz token limit very low
    )

    @agent
    def DataGatherer(self) -> Agent:
        return Agent(
            config=self.agents_config['DataGatherer'],
            verbose=True,
            inject_date=True,
            tools=[self.duck_search, self.yfin_search, self.website_scrpe],
            llm=self.gemini_lite
        )

    @agent
    def MarginOfSafetyInvestor(self) -> Agent:
        return Agent(
            config=self.agents_config['MarginOfSafetyInvestor'], # type: ignore[index]
            verbose=True,
            inject_date=True,
            llm=self.llama_big_new
        )

    @agent
    def HighGrowthInvestor(self) -> Agent:
        return Agent(
            config=self.agents_config['HighGrowthInvestor'], # type: ignore[index]
            verbose=True,
            inject_date=True,
            llm=self.llama_big_new
        )

    @agent
    def FinalResponseGenerator(self) -> Agent:
        return Agent(
            config=self.agents_config['FinalResponseGenerator'], # type: ignore[index]
            verbose=True,
            inject_date=True,
            llm=self.gemini
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def data_gathering_task(self) -> Task:
        return Task(
            config=self.tasks_config['data_gathering_task'], # type: ignore[index]
            output_file=str(self.PROJECT_ROOT/'tests'/'data.md')
        )

    @task
    def value_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['value_analysis_task'], # type: ignore[index]
            output_file=str(self.PROJECT_ROOT/'tests'/'value.md'),
            async_execution=False, ###### COULD BE TRUE BUT HITS MAX TOKEN PER MIN
            context=[self.data_gathering_task()], # for cleanliness cuz did it below. No it dosn't call function again cuz () used, @task does some magic
            callback=token_limit_saver
        ) 
    @task
    def growth_synthesis_task(self) -> Task:
        return Task(
            config=self.tasks_config['growth_synthesis_task'], # type: ignore[index]
            output_file=str(self.PROJECT_ROOT/'tests'/'growth.md'),
            async_execution=False, ########## SAME PROBLEM
            context=[self.data_gathering_task()] # cuz by default serial execution looks at task above and feed its output to next one
        )
    @task
    def final_response_task(self) -> Task:
        return Task(
            config=self.tasks_config['final_response_task'], # type: ignore[index]
            output_file=str(self.PROJECT_ROOT/'tests'/'verdict.md'),
            context=[self.value_analysis_task(), self.growth_synthesis_task()] # we we want two not just one of them
        )
    @crew
    def crew(self) -> Crew:
        """Creates the StockExpert crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            knowledge_sources=[self.investment_books],
            embedder={
                "provider": "sentence-transformer",
                "config": {
                    "model_name": "all-MiniLM-L6-v2"
                }
            }
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
