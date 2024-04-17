from src.gen_ai_tooling import Chain, Agent, Tools
from datetime import datetime
from pprint import pprint


def plan_trip(user_input: str):
    chain_prompt = [("system", """You are a holiday planning and suggestion agent, take into account users preferences and suggest a holiday destination. Output your answer in the format only outputting the destination and dates in the following format:
           Destination: destination_name
           Dates: holiday_start_date - holiday_end_date
           Reason: reason_for_suggesting_destination
           Original user query: {input}
           for reference current date and time is {current_date_time}"""),
        ("user", "{input}")]

    chain = Chain(chain_prompt)
    chain_response = chain.run_chain({"input": user_input, "current_date_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")})
    print(chain_response)
    
    
    agent_prompt = [("system", "You are very powerful travel plaining assistant."),
                ("system", """Your job is to plan a holiday based on the user input destination and dates.
                 Use relevant tools to find places to visit, accommodation options, and provide a weather report.
                 Output detailed itinerary and weather report for all days and clothes recommendations.
                 As a last step, always do a vector db similarity search through our internal recommendations to supplement the final answer.
                 """),
        ("user", "{input}")]

    agent = Agent(agent_prompt, [Tools.retriever_tool, Tools.get_weather, Tools.search_tool])

    agent_response = agent.run_agent({"input": chain_response})
    pprint(agent_response["output"])
    

def main_menu():
    while True:
        print("\nMain Menu")
        print("1. Plan a new trip")
        print("2. Exit")
        choice = input("Enter your choice (1 or 2): ")

        if choice == "1":
            plan_trip(input("Tell me about your dream holiday: "))
        elif choice == "2":
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice. Please enter 1 or 2.")

if __name__ == "__main__":
    main_menu()