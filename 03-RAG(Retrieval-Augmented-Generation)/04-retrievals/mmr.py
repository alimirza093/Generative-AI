from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_mistralai import MistralAIEmbeddings
from dotenv import load_dotenv


load_dotenv()

docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]

embed_model = MistralAIEmbeddings(model = "mistral-embed")

vectorStore = Chroma.from_documents(
    documents = docs,
    embedding = embed_model,
    persist_directory = "chroma_db"
)

print("\n===== Similarity Search Results =====\n")

similarity_retriever = vectorStore.as_retriever(
    search_type = "similarity",
    search_kwargs = {"k":3}
)

similarity_docs = similarity_retriever.invoke("What is gradient descent?")


for docs in similarity_docs:
    print(docs.page_content)
    
    
print("\n===== MMR Results =====\n")

mmr_retriever = vectorStore.as_retriever(
    search_type = "mmr",
    search_kwargs = {"k":3}
)

mmr_docs = mmr_retriever.invoke("What is gradient descent?")

for docs in mmr_docs:
    print(docs.page_content)