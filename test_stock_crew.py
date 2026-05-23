from helper_agent.crews.stock_expert.stock_expert import StockExpert

# We just need to call kickoff to test! how convinient.
result = StockExpert().crew().kickoff(inputs={"user_question" : "Should i buy more if a stock i bought went down sixteen percent.", "stock" : 'ITC'})
print(result)