from dotenv import load_dotenv
from pathlib import Path
from collections import defaultdict
import os
import ast

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from langchain_qdrant import QdrantVectorStore
from langchain_core.prompts import ChatPromptTemplate

# ----------------------------------
# Load environment variables
# ----------------------------------

load_dotenv()

# ----------------------------------
# Load PDF
# ----------------------------------

pdf_path = Path(__file__).parent / "mypdf.pdf"

loader = PyPDFLoader(str(pdf_path))
docs = loader.load()

# ----------------------------------
# Split documents
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
# Comment this block after first run
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
# Generate Sub Queries
# ----------------------------------

sub_query_prompt = ChatPromptTemplate.from_template("""
You are an expert query generator.

Generate 3 different search queries for the user question
to improve retrieval from a vector database.

Return ONLY a Python list.

Example:

[
"query 1",
"query 2",
"query 3"
]

Question:
{question}
""")

sub_query_chain = sub_query_prompt | llm

response = sub_query_chain.invoke({
    "question": question
})

print("\nSub Queries:")
print(response.content)

sub_queries = ast.literal_eval(response.content)

# ----------------------------------
# RAG Fusion Retrieval
# ----------------------------------

fused_scores = defaultdict(float)
doc_store = {}

k_rrf = 60

for query in sub_queries:

    docs = retriever.similarity_search(
        query=query,
        k=5
    )

    print(f"\nRetrieved for: {query}")

    for rank, doc in enumerate(docs):

        print(f"Rank {rank+1}")

        key = doc.page_content

        doc_store[key] = doc

        fused_scores[key] += 1 / (k_rrf + rank + 1)

# ----------------------------------
# Sort by RRF score
# ----------------------------------

reranked_results = sorted(
    fused_scores.items(),
    key=lambda x: x[1],
    reverse=True
)

# ----------------------------------
# Top Fused Documents
# ----------------------------------

top_docs = []

for doc_text, score in reranked_results[:5]:

    top_docs.append(doc_store[doc_text])

print("\nTop Ranked Chunks:\n")

for doc in top_docs:

    print("--------------------------------")
    print(doc.page_content)

# ----------------------------------
# Build Context
# ----------------------------------

context = "\n\n".join(
    [doc.page_content for doc in top_docs]
)

# ----------------------------------
# Final Prompt
# ----------------------------------

prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

Answer ONLY from the given context.

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
# Final Answer
# ----------------------------------

final_response = llm.invoke(final_prompt)

print("\nAnswer:\n")
print(final_response.content)