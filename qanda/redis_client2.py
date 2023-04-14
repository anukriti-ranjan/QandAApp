import pandas as pd
import pickle
import os
from dotenv import load_dotenv
import redis
from sentence_transformers import SentenceTransformer
import numpy as np
from redis.commands.search.query import Query
from redis.commands.search.field import (
    VectorField,
    NumericField,
    TextField,
    TagField
)
from langchain.vectorstores.redis import Redis as RedisVectorStore

## reference: https://lablab.ai/t/efficient-vector-similarity-search-with-redis-a-step-by-step-tutorial
## reference2: https://redis.com/blog/build-ecommerce-chatbot-with-redis/
NUMBER_DOCS = 2789
ITEM_DOC_EMBEDDING_FIELD = "embedding"
TEXT_EMBEDDING_DIMENSION=384

def setup_vector_db(redis_conn):


    docs = pd.read_csv("data/lte_docs_segmented.csv")
    docs.reset_index(drop=False, inplace=True)
    docs.rename(columns={"content": "page_content", "index":"metadata"}, inplace = True)
    #docs['metadata'] = docs['metadata'].apply(lambda x: {'id': x})

    # get the first N products with non-empty docs
    docs_data = docs.head(NUMBER_DOCS).to_dict(orient='index')

    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')


    item_contents =  [docs_data[i]['page_content']  for i in docs_data.keys()]
    #item_metadata =  [docs_data[i]['metadata']  for i in docs_data.keys()]
    #load from pickle file to save time
    item_content_vectors = [ model.encode(sentence) for sentence in item_contents]

    def load_vectors(client:redis.Redis, docs_data, vector_dict, vector_field_name):
        p = client.pipeline(transaction=False)
        for index in docs_data.keys():    
            #hash key
            key='doc:'+ str(index)
            
            #hash values
            item_data = docs_data[index]
            item_doc_vector = vector_dict[index].astype(np.float32).tobytes()
            item_data[vector_field_name]=item_doc_vector
            
            # HSET
            p.hset(key,mapping=item_data)
                
        p.execute()

    def create_hnsw_index (redis_conn,vector_field_name,number_of_vectors, vector_dimensions=512, distance_metric='L2',M=40,EF=200):
        redis_conn.ft().create_index([
            VectorField(vector_field_name, "HNSW", {"TYPE": "FLOAT32", "DIM": vector_dimensions, "DISTANCE_METRIC": distance_metric, "INITIAL_CAP": number_of_vectors, "M": M, "EF_CONSTRUCTION": EF}),
            TextField("page_content") ,
            NumericField("metadata")       
        ])    





    #flush all data
    redis_conn.flushall()

    #return vectorstore

    #create flat index & load vectors
    create_hnsw_index(redis_conn, ITEM_DOC_EMBEDDING_FIELD,NUMBER_DOCS,TEXT_EMBEDDING_DIMENSION,'COSINE',M=40,EF=200)
    load_vectors(redis_conn,docs_data,item_content_vectors,ITEM_DOC_EMBEDDING_FIELD)

#error

#'id': 'doc:874', 'payload': None, 'vector_score': '0.30083835125', 'content': 