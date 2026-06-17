from dotenv import load_dotenv
from pathlib import Path
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate

# Load environment variables
load_dotenv()

# ----------------------------------
# Load PDF
# ----------------------------------

pdf_path = Path(__file__).parent / "mypdf.pdf"

loader = PyPDFLoader(str(pdf_path))
docs = loader.load()

# ----------------------------------
# Split Documents
# ----------------------------------

text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

split_docs = text_splitter.split_documents(docs)

# ----------------------------------
# Embeddings
# ----------------------------------

embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ----------------------------------
# Store in Qdrant (Run ONLY once)
# ----------------------------------

vector_store = QdrantVectorStore.from_documents(
    documents=split_docs,
    embedding=embedder,
    url="http://localhost:6333",
    collection_name="myCollection"
)

# ----------------------------------
# Connect Existing Collection
# ----------------------------------

retriever = QdrantVectorStore.from_existing_collection(
    embedding=embedder,
    url="http://localhost:6333",
    collection_name="myCollection"
)

# ----------------------------------
# Ask Question
# ----------------------------------

question = "What is the Core Stack?"

relevant_chunks = retriever.similarity_search(
    query=question,
    k=3
)

context = "\n\n".join([doc.page_content for doc in relevant_chunks])

print("Relevant Chunks:\n")
print(context)

# ----------------------------------
# Gemini LLM
# ----------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# ----------------------------------
# Prompt Template
# ----------------------------------

prompt = ChatPromptTemplate.from_template("""
You are a helpful AI Assistant.

Answer the question only from the provided context.

Context:
{context}

Question:
{question}
""")

final_prompt = prompt.invoke({
    "context": context,
    "question": question
})

# ----------------------------------
# Generate Response
# ----------------------------------

response = llm.invoke(final_prompt)

print("\nAnswer:\n")
print(response.content)