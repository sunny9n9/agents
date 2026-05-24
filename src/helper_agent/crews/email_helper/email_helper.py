from crewai import Agent, Crew, Process, Task, LLM
from crewai.project import CrewBase, agent, crew, task
from crewai.agents.agent_builder.base_agent import BaseAgent
from ...tools.fetch_mail import get_crew_gmail_tools
from .models import Email, Email_Buckets, Email_List
# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators

@CrewBase
class EmailHelper():
    """EmailHelper crew"""

    _gmail_tools = None  # cache

    # Handled by @CrewBase, do not worry about populating these
    agents: list[BaseAgent]
    tasks: list[Task]

    gemini = LLM(temperature=0.0,
                 model="gemini/gemini-3.5-flash") # response format arg in this is a different kind one
    # Where did self.agents_config[] come from? and how does it map to the yaml file(s) ?
    # Well the @CrewBase decorator automatically visits and reads the yaml and makes them for us.
    @agent
    def mail_assistant(self) -> Agent:
        if EmailHelper._gmail_tools is None: 
            EmailHelper._gmail_tools = get_crew_gmail_tools()
        return Agent(
            config=self.agents_config['mail_assistant'], # type: ignore[index]
            verbose=True,
            # llm=self.gemini,
            tools=self._gmail_tools,
            inject_date=True
        )

    @agent
    def mail_summarizer(self) -> Agent:
        return Agent(
            config=self.agents_config['mail_summarizer'], # type: ignore[index]
            verbose=True,
            tools=self._gmail_tools,
            llm=self.gemini,
        )
    
    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task

    ############# Taks order matter ##############
    @task
    def email_summarization(self) -> Task:
        return Task(
            config=self.tasks_config['email_summary'],
            output_file=r"src\helper_agent\debug_files\summarized_emails.txt",
            output_json=Email_List # Output Pydantic causing internal server error 500
        )
    @task
    def email_sorting(self) -> Task:
        return Task(
            config=self.tasks_config['email_sorter'], # type: ignore[index]
            output_file=r"src\helper_agent\debug_files\sorted_mails.txt",
            output_json=Email_Buckets
        )
    @crew
    def crew(self) -> Crew:
        """Creates the EmailHelper crew"""

        return Crew(
            agents=self.agents, # Automatically created by the @agent decorator
            tasks=self.tasks, # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
        )
