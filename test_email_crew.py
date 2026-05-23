from helper_agent.crews.email_helper.email_helper import EmailHelper

# We just need to call kickoff to test! how convinient.
result = EmailHelper().crew().kickoff(inputs={"num_sentence_summary" : 3})
print(result)