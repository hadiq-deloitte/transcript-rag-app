'''
This Python script uses the RAG model to turn a YouTube transcript into a query-answering tool. It identifies relevant sections from the transcript and generates accurate responses to user queries.
'''

from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from langchain.prompts import ChatPromptTemplate

template = """
Answer the question based on the context below. If you can't 
answer the question, reply "I don't know".

Context: {context}

Question: {question}
"""

prompt = ChatPromptTemplate.from_template(template)

question = ""

# [DRAFT]