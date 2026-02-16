from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from groq import Groq
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph.message import add_messages
from dotenv import load_dotenv
import os

# Load variables from .env into environment
load_dotenv()

# Get the key
api_key = os.getenv("API_KEY")

# Ensure the key exists
if not api_key:
    raise ValueError("❌ API_KEY not found in .env file")


client = Groq(api_key=api_key)


llm = ChatGroq(
    api_key=api_key,
    model="llama-3.1-8b-instant"
)


class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]
    
    
    
def chat_node(state: ChatState):
    messages = state['messages']
    
    response = llm.invoke(messages)
    
    return {'messages':[response]}


checkpointer = InMemorySaver()
graph = StateGraph(ChatState)

graph.add_node('chat_node',chat_node)

graph.add_edge(START,'chat_node')
graph.add_edge('chat_node',END)


chatbot = graph.compile(checkpointer=checkpointer)
chatbot

