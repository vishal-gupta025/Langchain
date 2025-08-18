from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(file_path='Social_Network_Ads.csv')

docs = loader.lazy_load()

for doc in docs:
    print(doc.page_content)
