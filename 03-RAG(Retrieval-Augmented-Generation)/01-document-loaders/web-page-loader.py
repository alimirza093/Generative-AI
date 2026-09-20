from langchain_community.document_loaders import WebBaseLoader


url = 'https://www.cricinfo.com/story/i-want-to-do-something-special-616623'
web = WebBaseLoader(url)
docs = web.load()

for i in docs:
    print(docs[0].metadata)
    print(docs[0].page_content)