import os
import sys

from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    os.path.dirname(os.path.abspath(sys.argv[0])) + "\\data\\dl-curriculum.pdf"
)

docs = loader.load()

print(len(docs))

print(docs[0].page_content)
print(docs[1].metadata)
