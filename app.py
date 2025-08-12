import os
import base64
from pathlib import Path
import streamlit as st
import google.generativeai as genai
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from pinecone import Pinecone, ServerlessSpec

# ==== PAGE CONFIG ====
st.set_page_config(page_title="Zuhair's Chatbot", layout="wide")

# ==== BACKGROUND IMAGE ====
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

# ==== API KEYS ====
os.environ["GOOGLE_API_KEY"] = "AIzaSyCEFPOlW4mpLuLj7UwElumUeamlkPnAFGE"
os.environ["PINECONE_API_KEY"] = "pcsk_5o44TY_JrEUDmgwBrVp7bCasCqvjtAbxcTA5t51eyjqqR8CDQ7MPcRMCPXFW9M7u2TcrVZ"

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# ==== GEMINI EMBEDDINGS ====
class GeminiEmbeddings(Embeddings):
    def embed_documents(self, texts):
        embeddings = []
        for text in texts:
            res = genai.embed_content(
                model="models/embedding-001",
                content=text,
                task_type="retrieval_document"
            )
            embeddings.append(res["embedding"])
        return embeddings

    def embed_query(self, text):
        res = genai.embed_content(
            model="models/embedding-001",
            content=text,
            task_type="retrieval_query"
        )
        return res["embedding"]

# ==== LOAD PDF DOCUMENTS ====
loader = PyPDFDirectoryLoader("/content/pdf")  # Change path if needed
documents = loader.load()

# ==== SPLIT DOCUMENTS ====
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)

# ==== PINECONE INIT ====
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index_name = "vectordb"
dimension = 768
metric = "cosine"

existing_indexes = [index.name for index in pc.list_indexes()]
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(cloud="aws", region="us-west-2")
    )

# ==== VECTOR STORE ====
embeddings = GeminiEmbeddings()
vectorstore = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)
retriever = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings
).as_retriever()

# ==== GEMINI LLM ====
llm_model = genai.GenerativeModel(model_name="gemini-2.5-flash")

# ==== CHATBOT FUNCTION ====
def format_prompt(query, retrieved_docs):
    context = "\n\n".join([doc.page_content for doc in retrieved_docs])
    prompt = f"""You are an intelligent assistant.

Use the following context to answer the question at the end.

Context:
{context}

Question: {query}

Answer:"""
    return prompt

def chatbot(query):
    docs = retriever.get_relevant_documents(query)
    prompt = format_prompt(query, docs)
    response = llm_model.generate_content(prompt)
    return response.text

# ==== SESSION STATE ====
if "chat_sessions" not in st.session_state:
    st.session_state.chat_sessions = {"Chat 1": []}
    st.session_state.active_chat = "Chat 1"
    st.session_state.chat_count = 1

# ==== SIDEBAR ====
st.sidebar.title("💬 Chats")
chat_names = list(st.session_state.chat_sessions.keys())
selected_chat = st.sidebar.radio("Select a chat:", chat_names, index=chat_names.index(st.session_state.active_chat))
st.session_state.active_chat = selected_chat

if st.sidebar.button("➕ New Chat"):
    st.session_state.chat_count += 1
    new_chat = f"Chat {st.session_state.chat_count}"
    st.session_state.chat_sessions[new_chat] = []
    st.session_state.active_chat = new_chat

# ==== TITLE ====
st.title("🤖 Zuhair's Chatbot")

# ==== DISPLAY CHAT ====
for sender, message in st.session_state.chat_sessions[st.session_state.active_chat]:
    with st.chat_message("user" if sender == "You" else "assistant"):
        st.markdown(message)

# ==== USER INPUT ====
user_input = st.chat_input("Type your message...")

if user_input:
    with st.chat_message("user"):
        st.markdown(user_input)

    try:
        bot_response = chatbot(user_input)
    except Exception as e:
        bot_response = f"Backend Error: {str(e)}"

    with st.chat_message("assistant"):
        st.markdown(bot_response)

    st.session_state.chat_sessions[st.session_state.active_chat].append(("You", user_input))
    st.session_state.chat_sessions[st.session_state.active_chat].append(("Bot", bot_response))

    # Update Chat Title
    current_chat = st.session_state.active_chat
    if current_chat.startswith("Chat ") and len(st.session_state.chat_sessions[current_chat]) == 2:
        new_title = user_input.strip()[:30]
        st.session_state.chat_sessions[new_title] = st.session_state.chat_sessions.pop(current_chat)
        st.session_state.active_chat = new_title
