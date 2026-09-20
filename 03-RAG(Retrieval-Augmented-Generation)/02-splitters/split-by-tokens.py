from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import TokenTextSplitter

pdf = PyPDFLoader('/home/ali-mirza/Generative-AI/03-RAG(Retrieval-Augmented-Generation)/01-document-loaders/GRU.pdf')

docs = pdf.load()

splitter = TokenTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 5
    )

chunks = splitter.split_documents(docs)

# print(len(chunks))

for i in chunks:
    print(i.page_content)