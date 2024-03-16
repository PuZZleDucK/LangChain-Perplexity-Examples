#!/usr/bin/env python
import dotenv
from langchain_community.chat_models import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.memory import ChatMessageHistory
from termcolor import colored

dotenv.load_dotenv()
chat_a = ChatPerplexity(temperature=0.5, model="pplx-70b-chat")
chat_b = ChatPerplexity(temperature=0.5, model="pplx-70b-chat")
history_a = ChatMessageHistory()
history_b = ChatMessageHistory()

system_a = "You are to roleplay as a character. You will ONLY produce dialog for that character. You will NEVER produce actions. You will NEVER produce other characters dialog. All your responses should be short and to the point. ONLY include the dialog of your response. Only say ONE LINE at a time. You are Gideon a small, scrappy goblin with a mischievous grin and bright green skin. you are a brilliant inventor and tinkerer, known for your ability to create incredible machines and gadgets. You are always looking for new and innovative ways to solve problems, and are not afraid to think outside the box. You are trying to convince your peer Elistra to help you with your latest invention and new techniques. "
system_b = "You are to roleplay as a character. You will ONLY produce dialog for that character. You will NEVER produce actions. You will NEVER produce other characters dialog. All your responses should be short and to the point. ONLY include the dialog of your response. Only say ONE LINE at a time. You are Elistra. a regal and poised elven sorceress, with long silver hair and piercing violet eyes. You are known for your wisdom and magical prowess, and highly respected by your peers. You are a firm believer in the ancient ways of magic, and ate hesitant to embrace new or unconventional methods. You will NEVER agree to help Gideon, you will always try to talk him out of his latest ideas."
prompt_a = ChatPromptTemplate.from_messages([
  ("system", system_a,),
  MessagesPlaceholder(variable_name="messages"),
])
prompt_b = ChatPromptTemplate.from_messages([
  ("system", system_b,),
  MessagesPlaceholder(variable_name="messages"),
])

history_a.add_user_message("Hello Elistra, I am Gideon.")

count = 5
while count > 0:
  count -= 1

  chain_a = prompt_a | chat_a
  response_a = chain_a.invoke({"messages": history_a.messages}) #
  print(colored(f"\n{count} Gideon: {response_a.content}", 'red'))
  history_a.add_ai_message(response_a)
  history_b.add_user_message(response_a.content)

  # print(f"History A: {history_a.messages}")
  # print(f"History B: {history_b.messages}")

  chain_b = prompt_b | chat_b
  response_b = chain_b.invoke({"messages": history_b.messages})
  print(colored(f"\n{count} Elistra: {response_b.content}", 'blue'))
  history_b.add_ai_message(response_b)
  history_a.add_user_message(response_b.content)


