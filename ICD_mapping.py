from elasticsearch import Elasticsearch
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

# ==== Query ====
query_text = input("Enter diagnosis: ")
query_emb = model.get_embeddings([query_text])[0].values

# Hybrid query: BM25 + Vector
response = es.search(
    index="icd_index",
    body={
        "size": 5,
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

for hit in response["hits"]["hits"]:
    print(f"Score: {hit['_score']:.4f} | Code: {hit['_source']['code']} | Name: {hit['_source']['name']}")
