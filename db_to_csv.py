import pandas as pd
from pymongo import MongoClient
from bson import ObjectId

def db_to_csv():
    # MongoDB 연결 정보 설정
    # mongo_uri = 'mongodb://your_mongo_username:your_mongo_password@your_mongo_host:your_mongo_port/your_mongo_db'
    mongo_uri = "mongodb://ies:6b5@localhost:27017/"
    client = MongoClient(mongo_uri)
    db = client["mbti_database"]  # 실제 사용하는 MongoDB의 데이터베이스명으로 변경
    collection = db["mbti_collection"]  # 실제 사용하는 컬렉션명으로 변경
    # MongoDB 데이터 가져오기
    data_from_mongo = collection.find({})  # 모든 문서를 가져옴
    
    # 데이터 가공 및 CSV로 저장
    csv_data = {'text': [], 'mbti': []}

    for document in data_from_mongo:
        for i, message in enumerate(document['user_chat']):
            csv_data['text'].append(message)
            csv_data['mbti'].append(document['user_mbti'].lower())

    df = pd.DataFrame(csv_data)
    df.index.name = 'id'

    return df

# print(df)

# CSV 파일로 저장
# df.to_csv('output.csv', index_label='id')
