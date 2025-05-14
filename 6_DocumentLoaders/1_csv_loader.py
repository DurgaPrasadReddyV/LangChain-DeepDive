import os
import sys

from langchain_community.document_loaders import CSVLoader

loader = CSVLoader(
    file_path=os.path.dirname(os.path.abspath(sys.argv[0]))
    + "\\data\\Social_Network_Ads.csv"
)


docs = loader.load()

print(len(docs))
print(docs[1])
