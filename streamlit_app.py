import streamlit as st
import base64
from pathlib import Path

# ==== Page Config ====
st.set_page_config(page_title="Zuhair's Chatbot", layout="wide")

# ==== Background ====
def add_bg_from_local(image_file):
    with open(image_file, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()
    st.markdown(
        f"""
        <style>
        .stApp {{
            background-image: url("data:image/jpg;base64,{encoded_string}");
            background-size: cover;
            background-position: center;
            background-repeat: no-repeat;
        }}
        </style>
        """,
        unsafe_allow_html=True
    )

add_bg_from_local("162.jpg")

# ==== Session State ====
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {"Chat 1": []}
    st.session_state.active_chat = "Chat 1"
    st.session_state.chat_count = 1

# ==== Sidebar - Chat List ====
st.sidebar.title("💬 Chats")
chat_names = list(st.session_state.chat_sessions.keys())

# Select active chat
selected_chat = st.sidebar.radio("Select a chat:", chat_names, index=chat_names.index(st.session_state.active_chat))
st.session_state.active_chat = selected_chat

# Create new chat
if st.sidebar.button("➕ New Chat"):
    st.session_state.chat_count += 1
    new_chat = f"Chat {st.session_state.chat_count}"
    st.session_state.chat_sessions[new_chat] = []
    st.session_state.active_chat = new_chat

# ==== Title ====
# st.markdown("<h1 style='text-align: left; color: white;'>🤖 Zuhair's Chatbot</h1>", unsafe_allow_html=True)
st.title("🤖 Zuhair's Chatbot")

# ==== Display Chat History ====
for sender, message in st.session_state.chat_sessions[st.session_state.active_chat]:
    with st.chat_message("user" if sender == "You" else "assistant"):
        st.markdown(message)

# ==== Chat Input ====
user_input = st.chat_input("Type your message...")

if user_input:
    # Show user message
    with st.chat_message("user"):
        st.markdown(user_input)

    # Import backend
    try:
        from chatbot_backend import chatbot
        bot_response = chatbot(user_input)
    except Exception as e:
        bot_response = f"Backend Error: {str(e)}"

    # Show bot response
    with st.chat_message("assistant"):
        st.markdown(bot_response)

    # Save conversation
    st.session_state.chat_sessions[st.session_state.active_chat].append(("You", user_input))
    st.session_state.chat_sessions[st.session_state.active_chat].append(("Bot", bot_response))

    # === Update Chat Title ===
    current_chat = st.session_state.active_chat
    if current_chat.startswith("Chat ") and len(st.session_state.chat_sessions[current_chat]) == 2:
        new_title = user_input.strip()[:30]  # Limit to 30 characters
        # Rename the key in session_state
        st.session_state.chat_sessions[new_title] = st.session_state.chat_sessions.pop(current_chat)
        st.session_state.active_chat = new_title
