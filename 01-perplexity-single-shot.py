#!/usr/bin/env python
import dotenv
from langchain_community.chat_models import ChatPerplexity
from langchain_core.prompts import ChatPromptTemplate

dotenv.load_dotenv()
# model="pplx-70b-online"
# model="codellama-70b-instruct"
# model="mistral-7b-instruct"
# model="mixtral-8x7b-instruct"
chat = ChatPerplexity(temperature=0, model="sonar-medium-online") # sonar-small-chat

system = "You are a helpful assistant."
human = "{input}"
prompt = ChatPromptTemplate.from_messages([("system", system), ("human", human)])

user_input = input(">: ")
chain = prompt | chat
response = chain.invoke({"input": user_input})

print(f"$: {response.content}")
