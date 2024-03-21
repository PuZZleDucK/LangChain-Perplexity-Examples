#!/usr/bin/env python
import dotenv
from termcolor import colored
from langchain_community.chat_models import ChatPerplexity
from langchain.chains import ConversationChain
from langchain.memory import ConversationBufferMemory
dotenv.load_dotenv()

llm = ChatPerplexity(temperature=0, model="pplx-70b-chat")
conversation = ConversationChain(llm=llm, memory=ConversationBufferMemory())

user_input = ""
while user_input != "exit"  :
  user_input = input(">: ")

  response = conversation.predict(input=user_input)

  print(colored(f"$: {response}", 'cyan'))
