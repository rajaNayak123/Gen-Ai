from dotenv import load_dotenv
from pathlib import Path
import os
import ast

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate

# Load env
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
# Store in Qdrant (run once)
# ----------------------------------

vector_store = QdrantVectorStore.from_documents(
    documents=split_docs,
    embedding=embedder,
    url=os.getenv("QDRANT_URL"),
    collection_name="myCollection"
)

# ----------------------------------
# Connect existing collection
# ----------------------------------

retriever = QdrantVectorStore.from_existing_collection(
    embedding=embedder,
    url=os.getenv("QDRANT_URL"),
    collection_name="myCollection"
)

# ----------------------------------
# User Question
# ----------------------------------

question = "What is the Core Stack?"

# ----------------------------------
# LLM
# ----------------------------------

llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    api_key=os.getenv("GROQ_API_KEY")
)

# ----------------------------------
# STEP 1: Generate Sub-Queries
# ----------------------------------

sub_query_prompt = ChatPromptTemplate.from_template("""
You are a query expansion system.

Break the user question into 3 different search queries
to improve retrieval from a vector database.

Return ONLY a Python list.

Question: {question}
""")

sub_query_chain = sub_query_prompt | llm

sub_queries_response = sub_query_chain.invoke({"question": question})

print("\nSub Queries:\n", sub_queries_response.content)

sub_queries = ast.literal_eval(sub_queries_response.content)

# ----------------------------------
# STEP 2: Parallel Retrieval (simple loop version)
# ----------------------------------

all_chunks = []

for q in sub_queries:
    chunks = retriever.similarity_search(query=q, k=3)
    all_chunks.extend(chunks)

# ----------------------------------
# STEP 3: Deduplicate chunks
# ----------------------------------

unique_chunks = list({chunk.page_content: chunk for chunk in all_chunks}.values())

# ----------------------------------
# STEP 4: Build context
# ----------------------------------

context = "\n\n".join([doc.page_content for doc in unique_chunks])

print("\nRetrieved Context:\n")
print(context)

# ----------------------------------
# Prompt Template
# ----------------------------------

prompt = ChatPromptTemplate.from_template("""
You are a helpful AI Assistant.

Answer ONLY using the provided context.

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
# STEP 5: Final LLM Response
# ----------------------------------

response = llm.invoke(final_prompt)

print("\nAnswer:\n")
print(response.content)