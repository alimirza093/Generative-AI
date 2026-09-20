from langchain_core.documents import Document
from langchain_community.vectorstores import Chroma
from langchain_classic.retrievers.multi_query import MultiQueryRetriever
from langchain_mistralai import MistralAIEmbeddings, ChatMistralAI
from dotenv import load_dotenv


load_dotenv()


docs = [
    Document(page_content="Gradient descent is an optimization algorithm used in machine learning."),
    Document(page_content="Gradient descent minimizes the loss function."),
    Document(page_content="Gradient descent is an optimization that minimizes the loss function."),
    Document(page_content="Neural networks use gradient descent for training."),
    Document(page_content="Support Vector Machines are supervised learning algorithms.")
]

embed_model = MistralAIEmbeddings(model="mistral-embed")
llm_model = ChatMistralAI(model="open-mistral-7b", temperature=0.7)

vectorStore = Chroma.from_documents(
    documents=docs,
    embedding=embed_model,
    persist_directory="chroma_db"
)

retriever = vectorStore.as_retriever(
    search_type = "mmr",
    search_kwargs = {"k":3}
)

multi_query_retriever = MultiQueryRetriever.from_llm(
    retriever=retriever,
    llm=llm_model
)

result = multi_query_retriever.invoke("Usages of Gradient Descent?")

for r in result:
    print("-"*5)
    print(r.page_content)
    

