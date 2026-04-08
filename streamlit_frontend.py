import streamlit as st
import uuid
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from langgraph_database_backend import (
    chatbot,
    retrieve_all_threads,
    ingest_pdf
)


# ---------- Utility functions ----------
def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    if thread_id not in [t["id"] for t in st.session_state.chat_threads]:
        st.session_state.chat_threads.append({
            "id": thread_id,
            "name": thread_id[:8],
            "auto_named": False
        })


def reset_chat():
    st.session_state.thread_id = generate_thread_id()
    add_thread(st.session_state.thread_id)
    st.session_state.message_history = []


def delete_thread(thread_id):
    st.session_state.chat_threads = [
        t for t in st.session_state.chat_threads if t["id"] != thread_id
    ]

    if st.session_state.thread_id == thread_id:
        reset_chat()


def auto_rename_thread(thread_id, first_message):
    for t in st.session_state.chat_threads:
        if t["id"] == thread_id and not t["auto_named"]:
            t["name"] = first_message[:40]
            t["auto_named"] = True
            break


def load_conversation(thread_id):
    state = chatbot.get_state(config={"configurable": {"thread_id": thread_id}})
    return state.values.get("messages", [])


# ---------- Session setup ----------
if "thread_id" not in st.session_state:
    st.session_state.thread_id = generate_thread_id()

if "message_history" not in st.session_state:
    st.session_state.message_history = []

if "chat_threads" not in st.session_state:
    st.session_state.chat_threads = retrieve_all_threads()
    
if "ingested_docs" not in st.session_state:
    st.session_state.ingested_docs = {}

existing_ids = {t["id"] for t in st.session_state.chat_threads}
if st.session_state.thread_id not in existing_ids:
    add_thread(st.session_state.thread_id)


# ---------- Sidebar ----------
st.sidebar.title("BatuniGPT")

if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()

thread_key = str(st.session_state.thread_id)
thread_docs = st.session_state.ingested_docs.setdefault(thread_key, {})

st.sidebar.subheader("Upload PDF")

uploaded_pdf = st.sidebar.file_uploader(
    "Upload a PDF for this chat",
    type=["pdf"]
)

if uploaded_pdf:
    if uploaded_pdf.name in thread_docs:
        st.sidebar.info(f"{uploaded_pdf.name} already indexed.")
    else:
        with st.sidebar.status("Indexing PDF...", expanded=True) as status:
            summary = ingest_pdf(
                uploaded_pdf.getvalue(),
                thread_id=thread_key,
                filename=uploaded_pdf.name
            )
            thread_docs[uploaded_pdf.name] = summary
            status.update(label="PDF indexed", state="complete", expanded=False)

st.sidebar.header("My Conversations")

for thread in st.session_state.chat_threads[::-1]:
    thread_id = thread["id"]
    col1, col2 = st.sidebar.columns([6, 1])

    if col1.button(thread["name"], key=f"open_{thread_id}"):
        st.session_state.thread_id = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            if isinstance(msg, HumanMessage):
                role = "user"
            else:
                role = "assistant"

            temp_messages.append({
                "role": role,
                "content": msg.content
            })

        st.session_state.message_history = temp_messages
        st.rerun()

    if col2.button("🗑", key=f"del_{thread_id}"):
        delete_thread(thread_id)
        st.rerun()


# ---------- Display Previous Messages ----------
for message in st.session_state.message_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------- User Input ----------
user_input = st.chat_input("Type here...")

if user_input:

    # Save user message
    st.session_state.message_history.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    auto_rename_thread(st.session_state.thread_id, user_input)

    full_reply = ""
    
    with st.chat_message("assistant"):
        status_container = st.empty()
        def ai_stream():
            for message_chunk, _ in chatbot.stream(
                {"messages": [HumanMessage(content=user_input)]},
                config={"configurable": {"thread_id": thread_key}},
                stream_mode="messages"
            ):

                # Tool message
                if isinstance(message_chunk, ToolMessage):
                    tool_name = getattr(message_chunk, "name", "tool")

                    if tool_name == "calculator":
                        status_container.info("🧮 Calculating...")
                    elif tool_name == "get_stock_price":
                        status_container.info("📈 Fetching stock price...")
                    elif tool_name == "search":
                        status_container.info("🔍 Searching...")
                    elif tool_name == "rag_tool":
                        status_container.info("📄 Reading document...")
                    else:
                        status_container.info(f"⚙️ Running {tool_name}...")

                # AI response streaming
                if isinstance(message_chunk, AIMessage):
                    yield message_chunk.content

            status_container.empty()

        full_reply = st.write_stream(ai_stream())
    # Save assistant reply once
    st.session_state.message_history.append({
        "role": "assistant",
        "content": full_reply
    })


print("successfully ran frontend")