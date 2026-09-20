from langchain_community.document_loaders import TextLoader


text = TextLoader('03-RAG(Retrieval-Augmented-Generation)/01-document-loaders/notes.txt')
docs = text.load()

print(docs)


# if data is big then
# for i in docs:
#     print("Metadata: ",i.metadata)
#     print("Page Content: ",i.page_content)

