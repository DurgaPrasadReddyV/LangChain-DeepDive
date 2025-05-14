import os
import sys

from dotenv import load_dotenv
from langchain_community.document_loaders import TextLoader
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

model = ChatGoogleGenerativeAI(
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    model=os.getenv("LLM_MODEL"),
    temperature=0,
)

prompt = PromptTemplate(
    template="Write a summary for the following poem - \n {poem}",
    input_variables=["poem"],
)

parser = StrOutputParser()

loader = TextLoader(
    os.path.dirname(os.path.abspath(sys.argv[0])) + "\\data\\cricket.txt",
    encoding="utf-8",
)

docs = loader.load()

print(type(docs))

print(len(docs))

print(docs[0].page_content)

print(docs[0].metadata)

chain = prompt | model | parser

print(chain.invoke({"poem": docs[0].page_content}))
