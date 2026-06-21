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

# ----------------------------------
# Load Environment Variables
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
# Embeddings
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

question = """
Explain the Core Stack including
frontend, backend and database.
"""

# ----------------------------------
# Query Decomposition
# ----------------------------------

decompose_prompt = ChatPromptTemplate.from_template("""
Break the following question into
3 smaller independent questions.

Return ONLY a Python list.

Question:
{question}
""")

decompose_chain = decompose_prompt | llm

response = decompose_chain.invoke({
    "question": question
})

print("\nSub Questions:\n")
print(response.content)

sub_queries = ast.literal_eval(response.content)

# ----------------------------------
# Sequential Query Decomposition
# ----------------------------------

previous_answer = ""

all_answers = []

answer_prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

Use ONLY the provided context.

Previous Information:
{previous_answer}

Retrieved Context:
{context}

Current Question:
{question}
""")

for query in sub_queries:

    # Retrieve documents

    docs = retriever.similarity_search(
        query=query,
        k=3
    )

    context = "\n\n".join([
        doc.page_content
        for doc in docs
    ])

    # Create prompt

    prompt = answer_prompt.invoke({

        "previous_answer": previous_answer,
        "context": context,
        "question": query

    })

    # Generate answer

    response = llm.invoke(prompt)

    current_answer = response.content

    print("\n------------------------")
    print("Question:")
    print(query)

    print("\nAnswer:")
    print(current_answer)

    all_answers.append(current_answer)

    # Pass answer to next step

    previous_answer += "\n" + current_answer

# ----------------------------------
# Final Synthesis
# ----------------------------------

combined_answers = "\n\n".join(all_answers)

final_prompt = ChatPromptTemplate.from_template("""
You are a helpful AI assistant.

Combine the following information
and answer the original question.

Original Question:
{question}

Information:
{answers}
""")

final_input = final_prompt.invoke({

    "question": question,
    "answers": combined_answers

})

final_response = llm.invoke(final_input)

# ----------------------------------
# Final Output
# ----------------------------------

print("\n========================")
print("FINAL ANSWER")
print("========================\n")

print(final_response.content)