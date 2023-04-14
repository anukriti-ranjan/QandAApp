from redis.commands.search.query import Query

ITEM_DOC_EMBEDDING_FIELD = "embedding"
TEXT_EMBEDDING_DIMENSION=384

def search_semantic_redis(redis_conn, query_vector, topK) :

    #prepare the query
    q = Query(f'*=>[KNN {topK} @{ITEM_DOC_EMBEDDING_FIELD} $vec_param AS vector_score]').sort_by('vector_score').paging(0,topK).return_fields('vector_score','page_content', 'metadata').dialect(2)
    params_dict = {"vec_param": query_vector}


    #Execute the query
    results = redis_conn.ft().search(q, query_params = params_dict)

    my_docs = [doc.page_content for doc in results.docs]
    #my_docs = [doc for doc in results.docs]

    return my_docs
    #return results