from crewai import Agent, Crew, Process, Task
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.knowledge.source.pdf_knowledge_source import PDFKnowledgeSource
from crewai.knowledge.knowledge_config import KnowledgeConfig
import pathlib

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class StockExpert():
    """StockExpert crew"""

    agents: list[BaseAgent]
    tasks: list[Task]

    CURRENT_DIR = pathlib.Path(__file__).resolve().parent
    PROJECT_ROOT = CURRENT_DIR.parent.parent.parent.parent
    knowledge_base = PROJECT_ROOT / "knowledge"

    # build knowledge base
    knowledge_books = knowledge_base / "Books" # <------ more text-y pdf(s)
    # knowledge_books=pathlib.Path("knowledge/WealthCreationMO") # <------- very presentation like.mi8 need different parser
    # knowledge_config = KnowledgeConfig()
    books_list = []
    for p in knowledge_books.rglob('*.pdf'):
        books_list.append(str(p.relative_to(knowledge_base)))
        print(books_list)

    investment_books = PDFKnowledgeSource(
        chunk_size=4000,
        chunk_overlap=400,
        file_paths=books_list
    )
    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools
    @agent
    def MarginOfSafetyInvestor(self) -> Agent:
        return Agent(
            config=self.agents_config['MarginOfSafetyInvestor'], # type: ignore[index]
            verbose=True
        )

    @agent
    def HighGrowthInvestor(self) -> Agent:
        return Agent(
            config=self.agents_config['HighGrowthInvestor'], # type: ignore[index]
            verbose=True
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task
    @task
    def value_analysis_task(self) -> Task:
        return Task(
            config=self.tasks_config['value_analysis_task'], # type: ignore[index]
            output_file='value.md'
        )

    @task
    def growth_synthesis_task(self) -> Task:
        return Task(
            config=self.tasks_config['growth_synthesis_task'], # type: ignore[index]
            output_file='growth.md'
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
