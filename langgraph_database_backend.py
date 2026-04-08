from langgraph.graph import StateGraph, START, END
from typing import TypedDict, Annotated
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from langgraph.checkpoint.sqlite import SqliteSaver

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun

from dotenv import load_dotenv
import os
import sqlite3
import requests

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings

import tempfile

import streamlit as st


# =========================
# 1. ENV + LLM SETUP
# =========================

load_dotenv()

try:
    api_key = st.secrets["API_KEY1"]
except Exception:
    api_key = os.getenv("API_KEY1")

if not api_key:
    raise ValueError("API_KEY1 not found")



llm = ChatGroq(
    api_key=api_key,
    model="llama-3.3-70b-versatile"  
)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

print(".env loaded")
# =========================
# 2. TOOLS
# =========================

_THREAD_RETRIEVERS = {}
_THREAD_METADATA = {}

def _get_retriever(thread_id):
    if thread_id and str(thread_id) in _THREAD_RETRIEVERS:
        return _THREAD_RETRIEVERS[str(thread_id)]
    return None


# 🔎 Web Search Tool
duckduckgo = DuckDuckGoSearchRun(region="us-en")

@tool
def search(query: str) -> str:
    """Search the web for current information."""
    return duckduckgo.run(query)

search.name = "search"


def ingest_pdf(file_bytes: bytes, thread_id: str, filename: str = None):
    if not file_bytes:
        raise ValueError("No file uploaded")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as temp:
        temp.write(file_bytes)
        temp_path = temp.name

    try:
        loader = PyPDFLoader(temp_path)
        docs = loader.load()

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )

        chunks = splitter.split_documents(docs)

        vector_store = FAISS.from_documents(chunks, embeddings)
        retriever = vector_store.as_retriever(search_kwargs={"k": 4})

        _THREAD_RETRIEVERS[str(thread_id)] = retriever
        _THREAD_METADATA[str(thread_id)] = {
            "filename": filename,
            "documents": len(docs),
            "chunks": len(chunks)
        }

        return {
            "filename": filename,
            "documents": len(docs),
            "chunks": len(chunks)
        }

    finally:
        os.remove(temp_path)


# 🧮 Calculator Tool
@tool
def calculator(first_num: float, second_num: float, operation: str) -> dict:
    """
    Perform arithmetic: add, sub, mul, div
    """
    try:
        if operation == "add":
            result = first_num + second_num
        elif operation == "sub":
            result = first_num - second_num
        elif operation == "mul":
            result = first_num * second_num
        elif operation == "div":
            if second_num == 0:
                return {"error": "Division by zero not allowed"}
            result = first_num / second_num
        else:
            return {"error": f"Unsupported operation '{operation}'"}

        return {
            "first_num": first_num,
            "second_num": second_num,
            "operation": operation,
            "result": result
        }

    except Exception as e:
        return {"error": str(e)}


# 📈 Stock Price Tool (Hardcoded API key as requested)
@tool
def get_stock_price(symbol: str) -> dict:
    """
    Fetch latest stock price for a symbol using Alpha Vantage.
    """
    url = (
        "https://www.alphavantage.co/query?"
        f"function=GLOBAL_QUOTE&symbol={symbol}&apikey=QRB59RORCKQ3BFK8"
    )

    response = requests.get(url)
    data = response.json()

    try:
        quote = data["Global Quote"]
        return {
            "symbol": symbol,
            "price": quote["05. price"],
            "latest_trading_day": quote["07. latest trading day"],
            "volume": quote["06. volume"]
        }
    except Exception:
        return {"error": "Invalid symbol or API limit reached", "raw": data}


static_tools = [search, calculator, get_stock_price]

print("tools successfully integrated")

# =========================
# 3. STATE
# =========================

class ChatState(TypedDict):
    messages: Annotated[list[BaseMessage], add_messages]


# =========================
# 4. NODES
# =========================

from langchain_core.messages import SystemMessage

def _make_rag_tool(thread_id: str):
    @tool("rag_tool")  # ← explicit name prevents schema drift between calls
    def rag_tool(query: str) -> dict:
        """Search the uploaded PDF document and return relevant text chunks. Use this for any question about the uploaded file."""
        retriever = _get_retriever(thread_id)
        if retriever is None:
            return {"error": "No document uploaded for this thread."}
        docs = retriever.invoke(query)
        return {
            "query": query,
            "context": [doc.page_content for doc in docs],
            "metadata": [doc.metadata for doc in docs],
            "source_file": _THREAD_METADATA.get(str(thread_id), {}).get("filename")
        }
    return rag_tool

def chat_node(state: ChatState, config=None):
    thread_id = None
    if config:
        thread_id = config.get("configurable", {}).get("thread_id")

    rag_tool = _make_rag_tool(thread_id)
    has_pdf = _get_retriever(thread_id) is not None

    # Only include rag_tool if a PDF is loaded — fewer tools = fewer model errors
    runtime_tools = static_tools + ([rag_tool] if has_pdf else [])
    runtime_llm = llm.bind_tools(runtime_tools, tool_choice="auto")

    pdf_hint = (
        "A PDF is loaded. Use rag_tool to answer any questions about it."
        if has_pdf else
        "No PDF is loaded."
    )

    system = SystemMessage(
        content=(
            "You may use these tools when helpful: search, calculator, "
            f"get_stock_price{', rag_tool' if has_pdf else ''}. PDF status: {pdf_hint}"
        )
    )

    response = runtime_llm.invoke(
        [system] + state["messages"],
        config=config
    )

    return {"messages": [response]}


def tool_node(state: ChatState, config=None):
    thread_id = None
    if config:
        thread_id = config.get("configurable", {}).get("thread_id")

    rag_tool = _make_rag_tool(thread_id)
    has_pdf = _get_retriever(thread_id) is not None
    runtime_tools = static_tools + ([rag_tool] if has_pdf else [])
    node = ToolNode(runtime_tools)
    return node.invoke(state, config=config)

# =========================
# 5. CHECKPOINTER
# =========================

conn = sqlite3.connect(
    database="chatbot.db",
    check_same_thread=False
)

checkpointer = SqliteSaver(conn=conn)

print("database checkpointer done")

# =========================
# 6. GRAPH
# =========================

graph = StateGraph(ChatState)

graph.add_node("chat_node", chat_node)
graph.add_node("tools", tool_node)

graph.add_edge(START, "chat_node")

graph.add_conditional_edges(
    "chat_node",
    tools_condition,
    {
        "tools": "tools",
        "__end__": END
    }
)

graph.add_edge("tools", "chat_node")

chatbot = graph.compile(checkpointer=checkpointer)
print("Graph compiled successfully")

# =========================
# 7. THREAD RETRIEVAL
# =========================

def retrieve_all_threads():
    seen = set()
    threads = []

    for checkpoint in checkpointer.list(None):
        thread_id = checkpoint.config["configurable"]["thread_id"]

        if thread_id not in seen:
            seen.add(thread_id)
            threads.append({
                "id": thread_id,
                "name": thread_id[:8],
                "auto_named": False
            })

    return threads
print("thread retrieval done")


print("successfully ran backend")