from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
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

class Person(BaseModel):
    name: str = Field(description="Name of the person")
    age: int = Field( gt=18 , description="Age of the person")
    city: str = Field(description="City where the person lives")

parser = PydanticOutputParser(pydantic_object=Person)
templete = PromptTemplate(
    template= "generate the name , age , city of frictional {place} person \n {format_instruction}" ,
    input_variables=['place'],
    partial_variables= {'format_instruction': parser.get_format_instructions()}
)

Chain = templete | model | parser

result = Chain.invoke({"place": "India"})

print(result)

#used when you want to generate a pydantic model output  or Structured Output from the model
#and you can do data validation