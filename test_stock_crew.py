from helper_agent.crews.stock_expert.stock_expert import StockExpert

# We just need to call kickoff to test! how convinient.
result = StockExpert().crew().kickoff(inputs={"user_question" : "Should i buy exit my central bank position if it has accumulated a loss of 20%", "stock" : 'Central Bank'})
print(result)