import os
import google.generativeai as genai
from langchain_pinecone import PineconeVectorStore
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings.base import Embeddings
from pinecone import Pinecone, ServerlessSpec

# Set API keys
os.environ["GOOGLE_API_KEY"] = "AIzaSyCEFPOlW4mpLuLj7UwElumUeamlkPnAFGE"
os.environ["PINECONE_API_KEY"] = "pcsk_5o44TY_JrEUDmgwBrVp7bCasCqvjtAbxcTA5t51eyjqqR8CDQ7MPcRMCPXFW9M7u2TcrVZ"

genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

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

loader = PyPDFDirectoryLoader("/content/pdf")  # Update with your actual path
documents = loader.load()

# Split into chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
text_chunks = text_splitter.split_documents(documents)

# Import correct client
from pinecone import Pinecone, ServerlessSpec

# Initialize client
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])

# Define index details
index_name = "vectordb"
dimension = 768  # Gemini embedding dimension
metric = "cosine"

# Check if index exists
existing_indexes = [index.name for index in pc.list_indexes()]

# Create index if not exists
if index_name not in existing_indexes:
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric,
        spec=ServerlessSpec(
            cloud="aws",  # or "gcp"
            region="us-west-2"  # make sure this matches your project
        )
    )

# Initialize vector store with Gemini embeddings
embeddings = GeminiEmbeddings()

vectorstore = PineconeVectorStore.from_documents(
    documents=text_chunks,
    embedding=embeddings,
    index_name=index_name
)

retriever = vectorstore.as_retriever()

# Initialize Gemini model (correct way)
llm_model = genai.GenerativeModel(model_name="gemini-2.5-flash")

from langchain_pinecone import PineconeVectorStore

retriever = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings  # your GeminiEmbeddings instance
).as_retriever()

from langchain_pinecone import Pinecone
retriever = Pinecone.from_existing_index(index_name="vectordb", embedding=GeminiEmbeddings()).as_retriever()

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
    # Step 1: Retrieve relevant chunks
    docs = retriever.get_relevant_documents(query)

    # Step 2: Format prompt
    prompt = format_prompt(query, docs)

    # Step 3: Get Gemini response
    response = llm_model.generate_content(prompt)

    return response.text



# query = "What is FAST ?"
# answer = chatbot(query)
# print(answer)

for m in genai.list_models():
  if 'generateContent' in m.supported_generation_methods:
    print(m.name)

print(f"Loaded {len(text_chunks)} chunks")
for i, doc in enumerate(text_chunks[:3]):
    print(f"\n--- Chunk {i+1} ---\n{doc.page_content[:500]}")

# docs = retriever.get_relevant_documents(query)
# print(f"Retrieved {len(docs)} documents")

# if not docs:
#     print("No relevant documents retrieved!")

# print(f"Loaded {len(text_chunks)} chunks")
# for i, doc in enumerate(text_chunks[:3]):
#     print(f"\n--- Chunk {i+1} ---\n{doc.page_content[:500]}")

# results = retriever.get_relevant_documents("FAST University")
# print(f"Retrieved {len(results)} results")
# for r in results:
#     print(r.page_content[:200])

# Expose chatbot function for import
__all__ = ['chatbot']