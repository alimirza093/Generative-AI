from langchain_community.document_loaders import PyPDFLoader

pdf = PyPDFLoader('03-RAG(Retrieval-Augmented-Generation)/01-document-loaders/GRU.pdf')
docs = pdf.load()

for i in docs:
    print("Metadata: ",i.metadata)
    print("Page Content: ",i.page_content)