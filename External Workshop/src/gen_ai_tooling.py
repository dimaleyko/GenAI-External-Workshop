import openai
from dotenv import load_dotenv
import os
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.output_parsers import StrOutputParser
from langchain.agents.format_scratchpad.openai_tools import format_to_openai_tool_messages
from langchain.agents.output_parsers.openai_tools import OpenAIToolsAgentOutputParser
from langchain.agents import AgentExecutor, tool, Tool
from langchain_community.utilities import GoogleSearchAPIWrapper
import requests

load_dotenv()
openai.api_key = os.getenv('OPENAI_API_KEY')

model_name = "gpt-4-turbo-preview"

chat_model = ChatOpenAI(
                model=model_name,
                temperature=0,
                max_tokens=2000,
                api_key=openai.api_key,
                )

embedding_model = OpenAIEmbeddings()

if not openai.api_key:
    raise ValueError("API key not found. Ensure your .env file contains OPENAI_API_KEY.")

class Retriever():
    def __init__(self):
        loader = PyPDFLoader("./documents/Iceland.pdf")
        pages = loader.load_and_split()
        self.embedding_model = embedding_model
        self.vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=self.embedding_model)
        self.vectorstore.from_documents(pages, self.embedding_model, persist_directory="./chroma_db")

    def similarity_search(self, query, n=5):
        return self.vectorstore.similarity_search(query, n)
    
    
class Tools():
    
    @tool("weather-tool")
    def get_weather(lat: float, lon: float) -> int:
        """Returns the weather for a given latitude and longitude."""
        url = f"https://api.openweathermap.org/data/3.0/onecall?lat={lat}&lon={lon}&metric=metric&appid={os.getenv('OPENWEATHER_API_KEY')}"
        try:
            response = requests.get(url)
            response.raise_for_status()
            data = response.json()
            return data
        except requests.exceptions.RequestException as e:
            print(e)
            return None
        
    retriever = Retriever()
        
    weather_tool = get_weather
    search_tool = Tool(
        name="google_search",
        description="Search Google for recent results.",
        func=GoogleSearchAPIWrapper().run,
    )
    retriever_tool = Tool(
        name="retriever",
        description="Search the documents for relevant information.",
        func=retriever.similarity_search
    )
    

class Chain():
    def __init__(self, prompt, model=chat_model, output_parser=StrOutputParser()):
        """
        prompt has to to be a list of tuples with the following structure as an example:
        [("system", "You are a holiday planning and suggestion agent, take into account users preferences and suggest a holiday destination."),
        ("user", "{user_message}"),]
        """
        prompt = ChatPromptTemplate.from_messages(prompt)
        self.chain = prompt | model | output_parser
        
    def run_chain(self, parameters):
        """
        Ensure that you provide a dictionary map for parameters that you have in your prompt. Example:
        {"user_message": "I want to go on a holiday with my family of 4, we like beaches and warm weather."}
        """
        return self.chain.invoke(parameters)
    


class Agent():
    def __init__(self, prompt, tools, model=chat_model):
        """
        prompt has to to be a list of tuples with the following structure as an example:
        [("system", "You are a holiday planning and suggestion agent, take into account users preferences and suggest a holiday destination."),
        ("user", "{user_message}"),]
        """
        prompt.append(MessagesPlaceholder(variable_name="agent_scratchpad"))
        prompt = ChatPromptTemplate.from_messages(prompt)
        llm_with_tools = model.bind_tools(tools)
        
        agent = (
                        {
                            "input": lambda x: x["input"],
                            "agent_scratchpad": lambda x: format_to_openai_tool_messages(
                                x["intermediate_steps"]
                            ),
                        }
                        | prompt
                        | llm_with_tools
                        | OpenAIToolsAgentOutputParser()
                    )
        self.agent = AgentExecutor(agent=agent, tools=tools, verbose=True)
        
    def run_agent(self, parameters):
        """
        Ensure that you provide a dictionary map for parameters that you have in your prompt. Example:
        {"input": "I want to go on a holiday with my family of 4, we like beaches and warm weather."}
        """
        return self.agent.invoke(parameters)