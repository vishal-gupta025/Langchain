from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain.output_parsers import StructuredOutputParser , ResponseSchema
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

schema = [
    ResponseSchema(name = 'fact_1' , description = 'fact 1 about the topic'),
    ResponseSchema(name = 'fact_2' , description = 'fact 2 about the topic'),
    ResponseSchema(name = 'fact_3' , description = 'fact 3 about the topic')
]

parser = StructuredOutputParser.from_response_schemas(schema)

templete = PromptTemplate(
    template= "give me a 3 fact about {topic} \n {format_instruction}" ,
    input_variables=["topic"],
    partial_variables= {'format_instruction': parser.get_format_instructions()}
)

Chain = templete | model | parser
result = Chain.invoke({"topic" : "Black holes"})
print(result)


# you can get result in json format because the model is trained to return predefined json format
# but you can not do data validation 