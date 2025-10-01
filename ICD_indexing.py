from elasticsearch import Elasticsearch
import pandas as pd
from vertexai.language_models import TextEmbeddingModel
import vertexai
import os
from google.oauth2.service_account import Credentials
credentials_path = "prusandbx-nprd-uat-kw1ozq-dcfe6900463a.json"
credentials = Credentials.from_service_account_file(credentials_path,
            scopes=["https://www.googleapis.com/auth/cloud-platform"])

PROJECT_ID = "prusandbx-nprd-uat-kw1ozq"
REGION = "asia-southeast1"
from elasticsearch.helpers import bulk

vertexai.init(project=PROJECT_ID, location=REGION, credentials=credentials)

es = Elasticsearch("http://localhost:9200")

if not es.indices.exists(index="icd_index"):
    es.indices.create(
        index="icd_index",
        body={
            "settings": {
                "number_of_shards": 1,
                "number_of_replicas": 0
            },
            "mappings": {
                "properties": {
                    "code": {"type": "keyword"},
                    "name": {"type": "text"},
                    "embedding": {
                        "type": "dense_vector",
                        "dims": 768  # text-multilingual-embedding-002 trả về 768 chiều
                    }
                }
            }
        }
    )

# Load data
df = pd.read_csv("icd_data.csv")

# Init Gemini embeddings
model = TextEmbeddingModel.from_pretrained("text-multilingual-embedding-002")

es.indices.put_settings(
    index="icd_index",
    body={"settings": {"index": {"refresh_interval": "-1"}}}
)


BATCH_SIZE = 250  # hoặc 256, tuỳ GPU quota

for i in range(0, len(df), BATCH_SIZE):
    batch_df = df.iloc[i:i+BATCH_SIZE]
    texts = batch_df["Tên bệnh"].tolist()
    embeddings = model.get_embeddings(texts=texts)

    actions = []
    for row, emb in zip(batch_df.itertuples(index=False), embeddings):
        action = {
            "_index": "icd_index",
            "_id": row._asdict()["STT"],
            "_source": {
                "code": row._asdict()["Mã"],
                "name": row._2,
                "embedding": emb.values
            }
        }
        actions.append(action)
    success, failed = bulk(es, actions)
    print(f"Successfully indexed {success} documents, failed to index {failed} documents")

es.indices.put_settings(
    index="icd_index",
    body={"settings": {"index": {"refresh_interval": "1s"}}}
)

es.indices.refresh(index="icd_index")