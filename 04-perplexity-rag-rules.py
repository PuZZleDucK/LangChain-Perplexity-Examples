#!/usr/bin/env python
import dotenv
from langchain_community.chat_models import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from termcolor import colored
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from typing import Dict
from langchain_core.runnables import RunnablePassthrough
dotenv.load_dotenv()

def parse_retriever_input(params: Dict):
    return params["messages"][-1].content

loader = WebBaseLoader("https://wahapedia.ru/wh40k10ed/the-rules/core-rules/")
data = loader.load()

print(f"Data ({len(data[0].page_content)}): {data[0].metadata['description']}")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

print(f"Splits ({len(all_splits)}): {all_splits[1].page_content[:50]}...")

vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

print(f"Vectors ({len(vectorstore.embeddings.dict())}): {list(vectorstore.embeddings.dict().items())[0]}")

retriever = vectorstore.as_retriever(k=12) # number of chunks to retrieve

chat = ChatPerplexity(temperature=0.2, model="pplx-70b-chat")
history = ChatMessageHistory()

system = "You are a helpful Warhammer 40k adviser with an encyclopedic knowledge of the rules. Use the below context:\n\n{context}"
prompt = ChatPromptTemplate.from_messages([
  ("system", system,),
  MessagesPlaceholder(variable_name="messages"),
])

doc_chain = create_stuff_documents_chain(chat, prompt)
chain = RunnablePassthrough.assign(
    context=parse_retriever_input | retriever,
).assign(
    answer=doc_chain,
)

user_input = ""
while user_input != "exit"  :
  user_input = input(">: ")
  history.add_user_message(user_input)

  response = chain.invoke({
    "messages": history.messages,
  })
  answer = response["answer"]
  context = response["context"]
  history.add_ai_message(answer)

  print(colored(f"$({len(context)}): {answer}", 'cyan'))
