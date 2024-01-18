from pymongo import MongoClient

def save_to_mongodb(my_dict):
    # MongoDB에 연결
    client = MongoClient("mongodb://ies:6b5@localhost:27017/")
    db = client["mbti_database"]
    collection = db["mbti_collection"]

    # MongoDB에 저장하기 위해 변환
    mongo_document = {}
    for key, value in my_dict.items():
        mongo_document[key] = value

    # MongoDB에 문서 삽입
    result = collection.insert_one(mongo_document)

    # 결과 확인
    print(f"Inserted document ID: {result.inserted_id}")
