from pathlib import Path
import os
import faiss
import pickle
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chat_models import ChatOpenAI
from loguru import logger

cachepath = Path(os.getenv("CACHE_DIR", Path.home() / ".cache" / "chatlocal"))
indexpath = str(cachepath / "scepa.index")
storepath = cachepath / "scepa.pkl"

index = faiss.read_index(indexpath)

with open(str(storepath), "rb") as f:
    store = pickle.load(f)

store.index = index

chain = RetrievalQAWithSourcesChain.from_chain_type(
    llm=ChatOpenAI(temperature=0), retriever=store.as_retriever()
)
def qa(question: str) -> str:
    result = chain({"question": question})
    return result


logger.success("Ready to chat!")

if __name__ == "__main__":
    while True:
        question = input("You: ")
        answer = qa(question)
        print(f"Bot: {answer}")