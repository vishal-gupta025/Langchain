from langchain_community.document_loaders import TextLoader
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
import os

load_dotenv()

HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
repo_id = "deepseek-ai/DeepSeek-R1"
llm = HuggingFaceEndpoint(
    repo_id=repo_id,
    task="text-generation",
    max_new_tokens=100,
    temperature=0.7,
    huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN
)
model = ChatHuggingFace(llm=llm)

parser = StrOutputParser()

loader = TextLoader("cricket.txt") 

documents = loader.load()

prompt = PromptTemplate(
    template="generate a summary of the following text: \n {text}",
    input_variables=['text'],
)

print("Loaded documents:", documents[0].page_content) 

chain = prompt | model | parser

result = chain.invoke({"text": documents[0].page_content})

print("Summary:", result)