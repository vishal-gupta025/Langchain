from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence , RunnableParallel , RunnablePassthrough ,RunnableBranch
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

prompt1 = PromptTemplate(
    template = "generate a short note on the {topic}",
    input_variables=['topic'],
)

prompt2 = PromptTemplate(
    template="summerize the follwing notes in less then 500 words: \n {notes}",
    input_variables=['notes'],
)

sqeuence = RunnableSequence(prompt1 , model , parser)

conditional_sequence = RunnableBranch(
    (lambda x : len(x.split()) > 500 , RunnableSequence(prompt2 , model , parser)),
    RunnablePassthrough(sqeuence)    
)

final_sequence = RunnableSequence(sqeuence, conditional_sequence)

result = final_sequence.invoke({"topic": "AI and its impact on society"})

print(result)