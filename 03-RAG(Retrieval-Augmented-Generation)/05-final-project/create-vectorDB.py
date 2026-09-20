from langchain_mistralai import MistralAIEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

load_dotenv()

pdf = PyPDFLoader('03-RAG(Retrieval-Augmented-Generation)/05-final-project/deeplearning.pdf')
docs = pdf.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 200
)

chunks = splitter.split_documents(docs)

embed_model = MistralAIEmbeddings(model="mistral-embed")

vectorStore = Chroma.from_documents(
    documents = chunks,
    embedding = embed_model,
    persist_directory="chroma_db"
)




