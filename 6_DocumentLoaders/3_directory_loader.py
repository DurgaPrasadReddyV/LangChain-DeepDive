import os
import sys

from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader

loader = DirectoryLoader(
    path=os.path.dirname(os.path.abspath(sys.argv[0])) + "\\data",
    glob="*.pdf",
    loader_cls=PyPDFLoader,
)

docs = loader.lazy_load()

for document in docs:
    print(document.metadata)
