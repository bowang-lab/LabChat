import os
import threading
import queue
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from langchain.chat_models import ChatOpenAI
from langchain.callbacks.manager import AsyncCallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.prompts import PromptTemplate
from langchain.chains.question_answering import load_qa_chain

from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings()
persist_directory = '' # Data directory
vector_store = Chroma(persist_directory=persist_directory, embedding_function=embeddings)

# Modify the template below as needed
template = """You are a bot, named the [bot name] having a conversation with a human.

Background info:

Given the following extracted parts of a long document and a question, create a final answer.

{context}

{chat_history}
Human: {human_input}
Bot Name:"""

pp = PromptTemplate(
    input_variables=["chat_history", "human_input", "context"], template=template
)

class ThreadedGenerator:
    def __init__(self):
        self.queue = queue.Queue()

    def __iter__(self):
        return self

    def __next__(self):
        item = self.queue.get()
        if item is StopIteration: raise item
        return item

    def send(self, data):
        self.queue.put(data)

    def close(self):
        self.queue.put(StopIteration)

class ChainStreamHandler(StreamingStdOutCallbackHandler):
    def __init__(self, gen):
        super().__init__()
        self.gen = gen

    def on_llm_new_token(self, token: str, **kwargs):
        self.gen.send(token)

def llm_thread(g, prompt, chat_history):
    try:
        llm = ChatOpenAI(temperature=0, streaming=True, model='gpt-3.5-turbo-16k', callback_manager=AsyncCallbackManager([ChainStreamHandler(g)]))
        chain = load_qa_chain(llm, chain_type="stuff", prompt=pp)

        docs = vector_store.similarity_search(prompt)
        resp = chain({"input_documents": docs, "human_input": prompt, "chat_history": chat_history}, return_only_outputs=True)
        ans = resp

    finally:
        g.close()

def chain(prompt, chat_history):
    g = ThreadedGenerator()
    threading.Thread(target=llm_thread, args=(g, prompt, chat_history)).start()
    return g

class ChainRequest(BaseModel):
    message: str
    chat_history: str

app = FastAPI()

# CORS configuration
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/chain")
async def _chain(request: ChainRequest):
    gen = chain(request.message, request.chat_history)
    return StreamingResponse(gen, media_type="text/plain")
