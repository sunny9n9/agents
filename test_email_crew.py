from helper_agent.crews.email_helper.email_helper import EmailHelper

# We just need to call kickoff to test! how convinient.
result = EmailHelper().crew().kickoff()
print(result)