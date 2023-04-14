import os
import redis
import numpy as np
import pandas as pd
import typing as t
import pickle
from sentence_transformers import SentenceTransformer
from dotenv import load_dotenv

## copied from: https://github.com/RedisVentures/redis-openai-qna/blob/main/app/qna/db.py

from redis.commands.search.query import Query
from redis.commands.search.indexDefinition import (
    IndexDefinition,
    IndexType
)
from redis.commands.search.field import (
    VectorField,
    NumericField,
    TextField
)


INDEX_NAME = "embedding-index"
NUM_VECTORS = 2789
PREFIX = "embedding"
VECTOR_DIM = 384
DISTANCE_METRIC = "COSINE"
ITEM_KEYWORD_EMBEDDING_FIELD = "embedding"

def create_index(redis_conn: redis.Redis):
    # Define schema

    content = TextField(name="content")
    index = NumericField(name="index")
    embedding = VectorField("embedding",
        "FLAT", {
            "TYPE": "FLOAT64",
            "DIM": VECTOR_DIM,
            "DISTANCE_METRIC": DISTANCE_METRIC,
            "INITIAL_CAP": NUM_VECTORS
        }
    )
    # Create index
    #redis_conn.ft(INDEX_NAME).create_index(
    redis_conn.ft().create_index(
        fields = [index, content, embedding],
        #definition = IndexDefinition(prefix=[PREFIX], index_type=IndexType.HASH)
    )

def process_doc(doc) -> dict:
    d = doc.__dict__
    if "vector_score" in d:
        d["vector_score"] = 1 - float(d["vector_score"])
    return d

def search_redis(
    redis_conn: redis.Redis,
    query_vector: t.List[float],
    return_fields: list = [],
    k: int = 5,
) -> t.List[dict]:
    """
    Perform KNN search in Redis.
    Args:
        query_vector (list<float>): List of floats for the embedding vector to use in the search.
        return_fields (list, optional): Fields to include in the response. Defaults to [].
        k (int, optional): Count of nearest neighbors to return. Defaults to 5.
    Returns:
        list<dict>: List of most similar documents.
    """
    query_vector = query_vector.astype(np.float32).tobytes()
    # Prepare the Query

    #prepare the query
    q = Query(f'*=>[KNN {k} @{ITEM_KEYWORD_EMBEDDING_FIELD} $vec_param AS vector_score]').sort_by('vector_score').paging(0,k).return_fields('vector_score','index','content').dialect(2)
    params_dict = {"vec_param": query_vector}
    results = redis_conn.ft().search(q, query_params = params_dict)
    # base_query = f'*=>[KNN {k} @embedding $vector AS vector_score]'
    # query = (
    #     Query(base_query)
    #      .sort_by("vector_score")
    #      .paging(0, k)
    #      .return_fields(*return_fields)
    #      .dialect(2)
    # )
    # print(base_query)
    # params_dict = {"vector": query_vector.tobytes()}
    # # Vector Search in Redis
    # #results = redis_conn.ft(INDEX_NAME).search(query, params_dict)
    # results = redis_conn.ft().search(query, params_dict)
    #return [process_doc(doc) for doc in results.docs]
    return [doc for doc in results.docs]

def list_docs(redis_conn: redis.Redis, k: int = NUM_VECTORS) -> pd.DataFrame:
    """
    List documents stored in Redis
    Args:
        k (int, optional): Number of results to fetch. Defaults to VECT_NUMBER.
    Returns:
        pd.DataFrame: Dataframe of results.
    """
    base_query = f'*'
    return_fields = ['content']
    query = (
        Query(base_query)
        .paging(0, k)
        .return_fields(*return_fields)
        .dialect(2)
    )
    #results = redis_conn.ft(INDEX_NAME).search(query)
    results = redis_conn.ft().search(query)
    #return [process_doc(doc) for doc in results.docs]
    return [doc for doc in results.docs]

def index_documents(redis_conn: redis.Redis, embeddings_lookup: dict, documents: list):
    """
    Index a list of documents in RediSearch.
    Args:
        embeddings_lookup (dict): Doc embedding lookup dict.
        documents (list): List of docs to set in the index.
    """
    # Iterate through documents and store in Redis
    # NOTE: use async Redis client for even better throughput
    pipe = redis_conn.pipeline(transaction=False)
    for i, doc in enumerate(documents):
        #key = f"{PREFIX}:{i}"
        key = f"{PREFIX}:{i}"
        embedding = embeddings_lookup[doc["index"]]
        doc[ITEM_KEYWORD_EMBEDDING_FIELD] = embedding.astype(np.float32).tobytes()
        pipe.hset(key, mapping = doc)
        if i % 150 == 0:
            pipe.execute()
    pipe.execute()

def load_documents(redis_conn: redis.Redis):
    # Load data
    docs = pd.read_csv("data/lte_docs_segmented.csv")
    docs.reset_index(inplace=True, drop=False)
    with open('data/embeddings2.pkl', 'rb') as f:
        embeds = pickle.load(f)
    
    max_dim = embeds.shape[1]
    embeds = {
           (doc_num): doc_vector for doc_num, doc_vector in enumerate(embeds)
    }
    print(f"Indexing {len(docs)} Documents")
    index_documents(
        redis_conn = redis_conn,
        embeddings_lookup = embeds,
        documents = docs.to_dict("records")
    )
    print("Redis Vector Index Created!")

def init():
    redis_conn = redis.Redis(
        host=os.getenv('REDIS_HOST', 'localhost'),
        #user = os.getenv('REDIS_USER'),
        port=os.getenv('REDIS_PORT', 6379),
        #password=os.getenv('REDIS_PASSWORD')
    )

    # Check index existence
    if redis_conn.exists('embedding:1'):
        #redis_conn.ft(INDEX_NAME).info()
        
        print(redis_conn.type('embedding:1'))
        print("vectors exist")
    else:
        print("vectors do not exist")
        print("Creating embeddings index")
        # Create index
        create_index(redis_conn)
        load_documents(redis_conn)
    return redis_conn

if __name__=="__main__":
    load_dotenv()
    port=os.getenv('REDIS_PORT')
    password=os.getenv('REDIS_PASSWORD')
    print(f"port: {port}; password: {password}")
    redis_conn = init()
    print(len(list_docs(redis_conn)))
    model2 = SentenceTransformer('paraphrase-MiniLM-L6-v2')
    my_docs = pd.read_csv("data/lte_docs_segmented.csv")
    my_query = my_docs["content"].values[100]
    print(f"length of query is {len(my_query)}")
    query_vector = model2.encode(my_query)
    query_vector = query_vector.reshape(1, -1)
    print(f"shape of query_vector is {query_vector.shape}")
    
    my_list = search_redis(
    redis_conn = redis_conn,
    query_vector = query_vector,
    k = 5,
    return_fields=["content"],
    )
    print(len(my_list))

    #redis_client = redis.Redis(host='localhost', port=6379, db=0)

    # retrieve the document and its embedding by key
    #document_key = 'document:1'
    embedding_key = 'embedding:1'
    print(redis_conn.exists('embedding:1'))
    print(redis_conn.type('embedding:1'))
    #document = redis_client.get(document_key)
    #embedding_check = redis_conn.get(embedding_key)
    embedding_check = redis_conn.hgetall('embedding:1')
    # decode the retrieved values if necessary
#    document = document.decode('utf-8')
#    embedding = embedding_check.decode('utf-8')

    # print the retrieved values
#    print(document)
    #print(embedding_check)
    index_info = redis_conn.execute_command('FT.INFO')
    print(index_info)

# File "/home/.local/lib/python3.10/site-packages/streamlit/runtime/scriptrunner/script_runner.py", line 565, in _run_script
#     exec(code, module.__dict__)
# File "/home/QandABot/QandAApp/app.py", line 57, in <module>
#     st.write(chain.run(input_documents=search_results, question=inp_query))
# File "/home/.local/lib/python3.10/site-packages/langchain/chains/base.py", line 216, in run
#     return self(kwargs)[self.output_keys[0]]
# File "/home/.local/lib/python3.10/site-packages/langchain/chains/base.py", line 116, in __call__
#     raise e
# File "/home/.local/lib/python3.10/site-packages/langchain/chains/base.py", line 113, in __call__
#     outputs = self._call(inputs)
# File "/home/.local/lib/python3.10/site-packages/langchain/chains/combine_documents/base.py", line 56, in _call
#     output, extra_return_dict = self.combine_docs(docs, **other_keys)
# File "/home/.local/lib/python3.10/site-packages/langchain/chains/combine_documents/stuff.py", line 87, in combine_docs
#     inputs = self._get_inputs(docs, **kwargs)
# File "/home/.local/lib/python3.10/site-packages/langchain/chains/combine_documents/stuff.py", line 62, in _get_inputs
#     base_info = {"page_content": doc.page_content}
