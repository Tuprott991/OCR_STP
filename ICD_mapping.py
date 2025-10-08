from elasticsearch import Elasticsearch
from sqlalchemy import true
from vertexai.language_models import TextEmbeddingModel
import vertexai
from google.oauth2.service_account import Credentials

# ==== Setup Vertex AI + ES ====
credentials_path = "prusandbx-nprd-uat-kw1ozq-dcfe6900463a.json"
credentials = Credentials.from_service_account_file(credentials_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"])

PROJECT_ID = "prusandbx-nprd-uat-kw1ozq"
REGION = "asia-southeast1"

vertexai.init(project=PROJECT_ID, location=REGION, credentials=credentials)

es = Elasticsearch("http://localhost:9200")

model = TextEmbeddingModel.from_pretrained("text-multilingual-embedding-002")

def hybrid_search(query_text, top_k=5):
    response = es.search(
    index="icd_index",
    body={
        "size": top_k,
        "query": {
            "bool": {
                "should": [
                    {   # BM25 text search
                        "match": {
                            "name": query_text
                        }
                    },
                    {   # Semantic vector search
                        "script_score": {
                            "query": {"match_all": {}},
                            "script": {
                                "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                                "params": {"query_vector": query_emb}
                            }
                        }
                    }
                ]
            }
        }
    }
) 
    return response

def get_embedding(text):
    embedding = model.get_embeddings([text])[0].values
    print(embedding)
    return embedding

# Use levenshtein distance for fuzzy search
def fuzzy_search(query_text, top_k=5):
    response = es.search(
        index="icd_index",
        body={
            "size": top_k,
            "query": {
                "fuzzy": {
                    "name": {
                        "value": query_text,
                        "fuzziness": "AUTO"
                    }
                }
            }
        }
    )
    return response


# ==== Query ====
while True:
    query_text = input("Enter diagnosis: ")
    if query_text.lower() in ["exit", "quit"]:
        break
    query_emb = get_embedding(query_text)

    # response = hybrid_search(query_text, top_k=10)
    response = hybrid_search(query_text, top_k=10)
    print("Top ICD search results:")

    for hit in response["hits"]["hits"]:
        print(f"Score: {hit['_score']:.4f} | Code: {hit['_source']['code']} | Name: {hit['_source']['name']}")
