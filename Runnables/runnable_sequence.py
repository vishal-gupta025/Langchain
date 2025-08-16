from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain.schema.runnable import RunnableSequence
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
    template = "generate a joke on the {topic}",
    input_variables=['topic'],
)

prompt2 = PromptTemplate(
    template="explain the joke: \n {joke}",
    input_variables=['joke'],
)

sequence = RunnableSequence(prompt1 , model , parser , prompt2 , model, parser)

result = sequence.invoke({"topic": "Python programming"})

print(result)