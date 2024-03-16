#!/usr/bin/env python
import dotenv
from langchain_community.chat_models import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory

dotenv.load_dotenv()
# chat = ChatPerplexity(temperature=0, model="pplx-70b-online")
# codellama-70b-instruct -- mistral-7b-instruct -- mixtral-8x7b-instruct
chat = ChatPerplexity(temperature=0, model="pplx-70b-chat") # sonar-small-chat
demo_ephemeral_chat_history = ChatMessageHistory()

system = "You are a helpful assistant. Answer all questions to the best of your ability."
prompt = ChatPromptTemplate.from_messages([
  ("system", system,),
  MessagesPlaceholder(variable_name="messages"),
])

user_input = ""
while user_input != "exit"  :
  user_input = input(">: ")
  demo_ephemeral_chat_history.add_user_message(user_input)

  chain = prompt | chat
  response = chain.invoke({"messages": demo_ephemeral_chat_history.messages})
  demo_ephemeral_chat_history.add_ai_message(response)

  print(f"$: {response.content}")

