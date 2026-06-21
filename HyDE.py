from dotenv import load_dotenv
from pathlib import Path
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate

# ----------------------------------
# Load Environment
# ----------------------------------

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
# Embedding Model
# ----------------------------------

embedder = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# ----------------------------------
# Store in Qdrant (Run Once)
# ----------------------------------

vector_store = QdrantVectorStore.from_documents(
    documents=split_docs,
    embedding=embedder,
    url=os.getenv("QDRANT_URL"),
    collection_name="myCollection"
)

# ----------------------------------
# Connect Existing Collection
# ----------------------------------

retriever = QdrantVectorStore.from_existing_collection(
    embedding=embedder,
    url=os.getenv("QDRANT_URL"),
    collection_name="myCollection"
)

# ----------------------------------
# LLM
# ----------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# ----------------------------------
# User Question
# ----------------------------------

question = "What is the Core Stack?"

# ----------------------------------
# STEP 1: Generate Hypothetical Document
# ----------------------------------

hyde_prompt = ChatPromptTemplate.from_template("""
Write a detailed passage that answers
the following question.

Do not say you don't know.
Write a plausible answer.

Question:
{question}
""")

hyde_input = hyde_prompt.invoke({
    "question": question
})

hyde_response = llm.invoke(hyde_input)

hypothetical_doc = hyde_response.content

print("\nHypothetical Document:\n")
print(hypothetical_doc)

# ----------------------------------
# STEP 2: Retrieve using Hypothetical Document
# ----------------------------------

retrieved_docs = retriever.similarity_search(
    query=hypothetical_doc,
    k=5
)

# ----------------------------------
# STEP 3: Build Context
# ----------------------------------

context = "\n\n".join([
    doc.page_content
    for doc in retrieved_docs
])

print("\nRetrieved Context:\n")
print(context)

# ----------------------------------
# STEP 4: Final Answer
# ----------------------------------

final_prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

Answer ONLY using the provided context.

Context:
{context}

Question:
{question}
""")

final_input = final_prompt.invoke({

    "context": context,
    "question": question

})

final_response = llm.invoke(final_input)

print("\nFinal Answer:\n")

print(final_response.content)