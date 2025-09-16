# from youtube_transcript_api import YouTubeTranscriptApi

# vedio_url = "J5_-l7WIO_w"

# transcript_list = YouTubeTranscriptApi.get_transcript(vedio_url, languages=['hi'])
# transcript = " ".join(chunk["text"] for chunk in transcript_list)
# print(transcript)

from langchain.text_splitter import RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter.