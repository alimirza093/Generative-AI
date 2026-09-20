from langchain_community.vectorstores import Chroma
from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from dotenv import load_dotenv

load_dotenv()

embed_model = MistralAIEmbeddings(model="mistral-embed")
llm_model = ChatMistralAI(model="open-mistral-7b", temperature=0.7)

vectorStore = Chroma(
    persist_directory="chroma_db",
    embedding_function=embed_model
)


retriever = vectorStore.as_retriever(
    search_type = "mmr",
    kwargs = {
        "k":4,
        "fetch_k":10,
        "lambda_mult":0.5
    }
)

prompt_template = ChatPromptTemplate.from_messages(
     [(
          "system",
            """You are a helpful AI assistant.

Use ONLY the provided context to answer the question.

If the answer is not present in the context,
say: "I could not find the answer in the document."
"""
        ),
    ("human","""
     Context: {context}
     Question :{question}
     """)]
)


print("Rag System Created")
print("Press 0 to Exit" )
print("Ask any Question About PDF")
while True:
    prompt = input("you: ")
    if prompt == "0":
        break
    docs = retriever.invoke(prompt)
    context = "\n\n".join([doc.page_content for doc in docs])
    
    final_prompt = prompt_template.invoke({
        "context":context,
        "question":prompt
    })
    
    response = llm_model.invoke(final_prompt)
    
    print(response.content)