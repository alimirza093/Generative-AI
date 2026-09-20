from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

pdf = PyPDFLoader('/home/ali-mirza/Generative-AI/03-RAG(Retrieval-Augmented-Generation)/01-document-loaders/GRU.pdf')
docs = pdf.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 10
)

chunks = splitter.split_documents(docs)

# print(len(chunks))

print(chunks[1].page_content)



