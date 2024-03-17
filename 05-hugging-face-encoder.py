#!/usr/bin/env python
import dotenv
from termcolor import colored
from langchain_community.chat_models import ChatPerplexity
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import WebBaseLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage
dotenv.load_dotenv()

embeddings = HuggingFaceEmbeddings()

# prompt guides
# https://stable-diffusion-art.com/prompt-guide
# https://www.fotor.com/blog/stable-diffusion-prompts/
# https://nightcafe.studio/blogs/info/stable-diffusion-prompt-guide
# https://strikingloo.github.io/stable-diffusion-vs-dalle-2
loader = WebBaseLoader("http://anakin.ai/blog/stable-diffusion-prompt-guide")
data = loader.load()
print(colored(f"Data ({len(data[0].page_content)}): {data[0].metadata['description'][:50]}", 'blue', 'on_black', ['dark']))

query_result = embeddings.embed_query(data[0].page_content)
print(colored(f"Query result: ({len(query_result)}): {query_result[:3]}...", 'blue', 'on_black'))

doc_result = embeddings.embed_documents([data[0].page_content])
print(colored(f"Doc result: ({len(doc_result[0])}): {doc_result[0][:3]}...", 'blue', 'on_black', ['bold']))

chat = ChatPerplexity(temperature=0, model="pplx-70b-chat")

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)
vectorstore = Chroma.from_documents(documents=all_splits, embedding=embeddings)
print(colored(f"Vectors ({len(vectorstore.embeddings.dict())}): {list(vectorstore.embeddings.dict().items())[0][0]}...", 'cyan', 'on_black', ['dark']))

retriever = vectorstore.as_retriever(k=16)

system = """
Answer the user's questions based on the below context.
If the context doesn't contain any relevant information to the question, don't make something up and just say "I don't know":
<context>
{context}
</context>
"""
question_answering_prompt = ChatPromptTemplate.from_messages([
  ("system", system,),
  MessagesPlaceholder(variable_name="messages"),
])

document_chain = create_stuff_documents_chain(chat, question_answering_prompt)

base_response = document_chain.invoke(
    {
        "context": [],
        "messages": [
            HumanMessage(content="Outline emphasis syntax. answer in 100 words or less")
        ],
    }
)

docs = retriever.invoke("How do I make a chocolate cake?")
print(colored(f"Unrelated Docs search:({len(docs)}): {docs[0].page_content[:50]}...", 'cyan', 'on_black'))
unknown_response = document_chain.invoke(
    {
        "context": docs,
        "messages": [
            HumanMessage(content="How do I make a chocolate cake?")
        ],
    }
)

docs = retriever.invoke("Outline emphasis syntax")
print(colored(f"Related Docs search:({len(docs)}): {docs[0].page_content[:50]}...", 'cyan', 'on_black'))
rag_response = document_chain.invoke(
    {
        "context": docs,
        "messages": [
            HumanMessage(content="Outline emphasis syntax. answer in 100 words or less")
        ],
    }
)

print(colored(f"Base Response: {base_response}", 'red', 'on_black', ['dark']))
print(colored(f"Unknown Response: {unknown_response}", 'red', 'on_black', ['bold']))
print(colored(f"RAG Response: {rag_response}", 'cyan', 'on_black', ['bold']))
