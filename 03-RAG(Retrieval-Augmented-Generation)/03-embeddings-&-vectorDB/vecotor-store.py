from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv

load_dotenv()

docs = [
    Document(page_content="Python is widely used in Artificial Intelligence.", metadata={"source": "AI_book"}),
    Document(page_content="Pandas is used for data analysis in Python.", metadata={"source": "DataScience_book"}),
    Document(page_content="Neural networks are used in deep learning.", metadata={"source": "DL_book"}),
]

embed_model = MistralAIEmbeddings(model="mistral-embed")

vectorStore = Chroma.from_documents(
    documents = docs,
    embedding = embed_model,
    persist_directory = "chroma_db"
)

result = vectorStore.similarity_search("What is the usages of Python", k=2)

for r in result:
    print("-"*8)
    print(r.page_content)