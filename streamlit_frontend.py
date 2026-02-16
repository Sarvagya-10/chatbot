import streamlit as st
import uuid
import time
from langchain_core.messages import HumanMessage, AIMessage
from langgraph_backend import chatbot


# ---------- Utility functions ----------
def generate_thread_id():
    return str(uuid.uuid4())


def add_thread(thread_id):
    if thread_id not in [t["id"] for t in st.session_state.chat_threads]:
        st.session_state.chat_threads.append({
            "id": thread_id,
            "name": thread_id[:8],  # temporary name
            "auto_named": False     # track if first message used
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
            t["name"] = first_message[:40]  # truncate title
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
    st.session_state.chat_threads = []

add_thread(st.session_state.thread_id)


# ---------- Sidebar UI ----------
st.sidebar.title("LangGraph Chatbot")

if st.sidebar.button("New Chat"):
    reset_chat()
    st.rerun()

st.sidebar.header("My Conversations")

for thread in st.session_state.chat_threads[::-1]:
    thread_id = thread["id"]

    col1, col2 = st.sidebar.columns([6, 1])

    # ---------- Open thread ----------
    if col1.button(thread["name"], key=f"open_{thread_id}"):
        st.session_state.thread_id = thread_id
        messages = load_conversation(thread_id)

        temp_messages = []
        for msg in messages:
            role = "user" if isinstance(msg, HumanMessage) else "assistant"
            temp_messages.append({"role": role, "content": msg.content})

        st.session_state.message_history = temp_messages
        st.rerun()

    # ---------- Delete ----------
    if col2.button("🗑", key=f"del_{thread_id}"):
        delete_thread(thread_id)
        st.rerun()


# ---------- Load previous messages ----------
for message in st.session_state.message_history:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])


# ---------- User input ----------
user_input = st.chat_input("Type here...")

if user_input:
    # store user message
    st.session_state.message_history.append({
        "role": "user",
        "content": user_input
    })

    with st.chat_message("user"):
        st.markdown(user_input)

    # 🔥 AUTO-RENAME on first message only
    auto_rename_thread(st.session_state.thread_id, user_input)

    # send ONLY new message to LangGraph
    messages = [HumanMessage(content=user_input)]

    response_container = st.empty()
    full_reply = ""

    for chunk, metadata in chatbot.stream(
        {"messages": messages},
        config={"configurable": {"thread_id": st.session_state.thread_id}},
        stream_mode="messages",
    ):
        if isinstance(chunk, AIMessage) and chunk.content:
            for char in chunk.content:
                full_reply += char
                response_container.chat_message("assistant").markdown(full_reply)
                time.sleep(0.001)

    # save assistant reply
    st.session_state.message_history.append({
        "role": "assistant",
        "content": full_reply
    })
