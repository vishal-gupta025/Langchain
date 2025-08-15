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

templete1 = PromptTemplate(
    template= "give a brief content about {topic}" ,
    input_variables=["topic"]
)

templete2 = PromptTemplate(
    template= " genreate a 5 points summery about {text}" ,
    input_variables=["text"]
) 

parser = StrOutputParser()  

Chain = templete1 | model | parser | templete2 | model | parser 

result = Chain.invoke({"topic" : "Black holes"})

print(result)

#used when you want to generate a string output from the model
#but you can not do data validation
