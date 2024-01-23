from pymongo import MongoClient
import os
from dotenv import load_dotenv

load_dotenv()

Mongo_uri=os.getenv('Mongo_uri')
Mongo_db=os.getenv('Mongo_db')
Mongo_new_collection=os.getenv('Mongo_new_collection')

def save_to_mongodb(my_dict):
    # MongoDB에 연결
    client = MongoClient(Mongo_uri)
    db = client[Mongo_db]
    collection = db[Mongo_new_collection]

    # MongoDB에 저장하기 위해 변환
    mongo_document = {}
    for key, value in my_dict.items():
        mongo_document[key] = value

    # MongoDB에 문서 삽입
    result = collection.insert_one(mongo_document)

    # 결과 확인
    print(f"Inserted document ID: {result.inserted_id}")
