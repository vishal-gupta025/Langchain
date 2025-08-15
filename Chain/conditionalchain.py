from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableBranch , RunnableLambda
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field
from typing import Literal
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

class feedback(BaseModel):
    sentiment: Literal["pos", "neg"] = Field(description="give the sentiment of the feedback")

parser1 = StrOutputParser()

parser2 = PydanticOutputParser(pydantic_object=feedback)

prompt1 = PromptTemplate(
    template = "classify the  following feedback as positive or negative: \n {feedback} \n {format_instruction}",
    input_variables=['feedback'],
    partial_variables={'format_instruction': parser2.get_format_instructions()}
)
classifire_chain = prompt1 | model | parser2

prompt2 = PromptTemplate(
    template = "write a appropriate response for this positive feedback: \n {feedback} ",
    input_variables=['feedback']
)

prompt3 = PromptTemplate(
    template = "write a appropriate response for this negative feedback: \n {feedback} ",
    input_variables=['feedback']
)
 
branch_chain = RunnableBranch(
    (lambda x: x.sentiment == "postive", prompt2 | model | parser1),
    (lambda x: x.sentiment == "negative", prompt3 | model | parser1),
    RunnableLambda(lambda x: "could not found sentiment")
)

chain = classifire_chain | branch_chain

result = chain.invoke({"feedback": "This product is amazing!"})

print(result)

chain.get_graph().print_ascii()