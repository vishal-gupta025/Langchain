from dotenv import load_dotenv
from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint , HuggingFaceEmbeddings 
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from youtube_transcript_api import YouTubeTranscriptApi , TranscriptsDisabled
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate

load_dotenv()

import os 
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

video_id = "LPZh9BOjkQs"

try :
    transcript_list = YouTubeTranscriptApi.get_transcript(video_id)
    print(transcript_list)
    # transcript = " ".join(chunk["text"] for chunk in transcript_list)
    # print(transcript)
except TranscriptsDisabled:
    print("Transcripts are disabled for this video.")

# print(transcript_list)