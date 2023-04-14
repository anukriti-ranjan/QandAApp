from dotenv import load_dotenv
import streamlit as st
import os
import unicodedata
import redis
import numpy as np
from sentence_transformers import SentenceTransformer
from qanda.redis_client2 import setup_vector_db
from qanda.helper_fn import search_semantic_redis
#from langchain.llms import OpenAI
#from langchain.chains.question_answering import load_qa_chain
#from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain.callbacks.base import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import (
    ConversationalRetrievalChain,
    LLMChain
)
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.prompts.prompt import PromptTemplate



# Load the environment variables from the .env file
load_dotenv()


redis_conn = redis.Redis(
    host=os.getenv('REDIS_HOST', 'localhost'),
    #user = os.getenv('REDIS_USER'),
    port=os.getenv('REDIS_PORT', 6379),
    #password=os.getenv('REDIS_PASSWORD')
)
print ('Connected to redis')
setup_vector_db(redis_conn)

model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
#llm = OpenAI(temperature=0, openai_api_key=os.getenv('OPENAI_API_KEY'))
#chain = load_qa_chain(llm, chain_type="stuff")
#chain = load_qa_with_sources_chain(llm, chain_type="stuff")

#query = " "

def clean_text(s):
    # Turn a Unicode string to plain ASCII
    def unicodeToAscii(s1):
        return ''.join(
            c for c in unicodedata.normalize('NFD', s1)
            if unicodedata.category(c) != 'Mn')
    #s = s.replace("&nbsp;","")
    s =" ".join(unicodeToAscii(s.strip()).split())
    return s.lower()

template = """You are a friendly, conversational assistant for 3gpp performance counters. 
Use the following context and answer any questions.
It's ok if you don't know the answer.
Context: {context} \n
Question: {query}
"""



# define two LLM models from OpenAI
llm = OpenAI(temperature=0)
 


st.header('Q and A App')


with st.form('search_form'):
    inp_query = st.text_input('Search your query',"")
    submitted = st.form_submit_button('Submit')
    if submitted:
        #vectorize the query
        query_vector = model.encode(inp_query).astype(np.float32).tobytes()

        #use redis to extract top 5 results
        search_results = search_semantic_redis(redis_conn, query_vector, 5)

        qa_prompt= PromptTemplate(template=template, input_variables=['context','query'])

        chain = LLMChain(llm=llm,prompt=qa_prompt)

        ques_context = ""

        for i,result in enumerate(search_results):
            ques_context = str(i) + ") :"+result+"\n"


        input = {'context':ques_context,'query':inp_query}
        st.write(chain.run(input))



        











