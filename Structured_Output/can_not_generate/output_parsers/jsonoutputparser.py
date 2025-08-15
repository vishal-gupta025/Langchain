from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import JsonOutputParser 
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

parser = JsonOutputParser()

templete = PromptTemplate(
    template= "give me a 5 fact about {topic} \n {format_instruction}" ,
    input_variables=["topic"],
    partial_variables= {'format_instruction': parser.get_format_instructions()} 
)

Chain = templete | model | parser

result = Chain.invoke({"topic" : "Black holes"})
print(result)

# you can't get result in json format because the model is not trained to return predefined json format