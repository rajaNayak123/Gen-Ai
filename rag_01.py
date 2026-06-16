from dotenv import load_dotenv
from pathlib import Path
import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import CharacterTextSplitter


load_dotenv()


pdf_path = Path(__file__).parent / "mypdf.pdf"

loader = PyPDFLoader(file_path=pdf_path)

docs = loader.load()


text_splitter = CharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

split_docs = text_splitter.split_documents(docs)