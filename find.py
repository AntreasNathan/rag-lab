import sys
from rag import load_docs, build_chunks
import re

term = sys.argv[1]
chunks = build_chunks(load_docs())

for chunk in chunks:
    if(chunk["text"].count("displaystyle") > 2):
        continue
    if(term.lower() in chunk["text"].lower()):
        text_pos = chunk['text'].lower().find(term.lower())
        text = chunk["text"][max(0,text_pos-100):min(text_pos+150, len(chunk["text"]))]
        text = re.sub(r"\s+", " ", text)
        print(f"Id: {chunk['id']} -- Text around: {text}")
