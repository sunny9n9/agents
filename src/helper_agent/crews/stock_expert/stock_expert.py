from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task, before_kickoff
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.tasks.task_output import TaskOutput
from crewai_tools import RagTool
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
from crewai.knowledge.knowledge_config import KnowledgeConfig
from ...tools.search_tool import yf_search, ddg_search, jina_search
# from crewai_tools import ScrapeWebsiteTool replaced with jina_search, lot better quality scraping.
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
    # books_list = []
    # for p in knowledge_books.rglob('*.pdf'):
    #     books_list.append(str(p.relative_to(knowledge_base)))
    #     print(books_list)

    # investment_books = PDFKnowledgeSource(
    #     chunk_size=1000,
    #     chunk_overlap=200,
    #     file_paths=books_list
    # )

    # -------------------------------------------------------------------- [Getting Tool(s)]
    duck_search = ddg_search()
    yfin_search = yf_search() # does NOT return ANYTHING useful for indian stocks
    website_scrpe = jina_search() # only returns non js stuff without links T-T
    rag_value = RagTool(
        name="Parag Parik Framework",
        description="A guide to investment in Indian stock market by Parag Parik",
        # summarize=True # but by how much, i still need details
        config={
            "embedding_model": {
                "provider": "sentence-transformer",
                "config": {
                    "model_name": "all-MiniLM-L6-v2",
                    "normalize_embeddings": True
                }
        }
    }
    )
    rag_growth = RagTool(
        name="Growth Investing Guide by Mukherjea",
        description="A guide to investments in Indian Stock market via Coffee Can Portfolio by Saurabh Mukherjea",
        config={
            "embedding_model": {
                "provider": "sentence-transformer",
                "config": {
                    "model_name": "all-MiniLM-L6-v2",
                    "normalize_embeddings": True
                }
        }
    }
    )
    ## Pdf are added by @before_kickoff() function
    # -------------------------------------------------------------------- [Agent Definitions]
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools

    gemini = LLM(
        model="gemini/gemini-2.5-flash", # <----- lord and saviour big context winodow
        temperature=0.4,
        reasoning_effort='medium', 
    )
    gemini_lite = LLM(
        model="gemini/gemini-2.5-flash-lite",
        temperature=0.4
    )
    llama_small = LLM(
        model="groq/llama-3.1-8b-instant",
        temperature=0.3 # <---- smaller one for very quick referencing
    )
    llama_big_old = LLM(
        model="groq/llama-3.1-70b-versatile",
        temperature=0.3 # <---- little older version, thought to trick groq token limit but doesn't work that way
    )
    llama_big_new = LLM(
        model="groq/llama-3.3-70b-versatile",
        temperature=0.3 # <---- copy of above cuz token limit very low
    )

    @before_kickoff
    def prepare_financial_libraries(self, inputs):
        """Executes safely immediately before the crew starts processing data."""
        try:
            # Perform mutations right before execution starts
            self.rag_growth.add(data_type="file", path=str(self.knowledge_books / "Coffee_Can_Saurabh_Mukherjea.pdf"))
            self.rag_value.add(data_type="file", path=str(self.knowledge_books / "Value_Investing_Parag_Parikh.pdf"))
        except Exception as e:
            print(f"[ERROR] :: Cannot add files to rag tool :: {e}")
        return inputs # must return inputs from before_kickoff hooks for kickoff 

    @agent
    def DataGatherer(self) -> Agent:
        return Agent(
            config=self.agents_config['DataGatherer'],
            verbose=True,
            inject_date=True,
            tools=[self.duck_search, self.yfin_search, self.website_scrpe],
            llm=self.llama_small
        )

    @agent
    def MarginOfSafetyInvestor(self) -> Agent:
        return Agent(
            config=self.agents_config['MarginOfSafetyInvestor'], # type: ignore[index]
            verbose=True,
            inject_date=True,
            tools=[self.rag_value],
            # allow_code_execution=True, # so it can correctly calculate whatever ratio(s) it might need, though it might be walking of knife in terms of context window size
                            # basically makes a CodeINterpreterTool() and adds it to the tool lists, better do it manually for control and expectiations alignment
                            # Oh Wait, its depreciated, both of them
            llm=self.llama_big_new
        )

    @agent
    def HighGrowthInvestor(self) -> Agent:
        return Agent(
            config=self.agents_config['HighGrowthInvestor'], # type: ignore[index]
            verbose=True,
            inject_date=True,
            tools=[self.rag_growth],
            # allow_code_execution=True, Depreciated, make a docker and pass it as tool
            llm=self.llama_big_new
        )

    @agent
    def PersonalPortfolioManager(self) -> Agent:
        return Agent(
            config=self.agents_config['PersonalPortfolioManager'], # type: ignore[index]
            verbose=True,
            inject_date=True,
            llm=self.llama_big_new
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def data_gathering_task(self) -> Task:
        return Task(
            config=self.tasks_config['data_gathering_task'], # type: ignore[index]
            output_file=str(self.PROJECT_ROOT/'src'/'helper_agent'/'debug_files'/'data.txt')
        )

    @task
    def value_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['value_analysis_task'], # type: ignore[index]
            output_file=str(self.PROJECT_ROOT/'src'/'helper_agent'/'debug_files'/'value.txt'),
            async_execution=False, ###### COULD BE TRUE BUT HITS MAX TOKEN PER MIN
            context=[self.data_gathering_task()], # for cleanliness cuz did it below. No it dosn't call function again cuz () used, @task does some magic
            callback=token_limit_saver
        ) 
    @task
    def growth_synthesis_task(self) -> Task:
        return Task(
            config=self.tasks_config['growth_synthesis_task'], # type: ignore[index]
            output_file=str(self.PROJECT_ROOT/'src'/'helper_agent'/'debug_files'/'growth.txt'),
            async_execution=False, ########## SAME PROBLEM
            context=[self.data_gathering_task()], # cuz by default serial execution looks at task above and feed its output to next one
            callback=token_limit_saver
        )
    @task
    def portfolio_management_task(self) -> Task:
        return Task(
            config=self.tasks_config['portfolio_management_task'], # type: ignore[index]
            output_file=str(self.PROJECT_ROOT/'src'/'helper_agent'/'debug_files'/'verdict.txt'),
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
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
