from pymongo import MongoClient
import pandas as pd

def ogcsv():
    # MongoDB에 연결
    client = MongoClient("mongodb://ies:6b5@localhost:27017/")
    db = client['input_database']
    collection = db['dataset_collection']

    # 컬렉션에서 데이터 조회
    cursor = collection.find()

    # 조회된 데이터를 리스트로 저장
    data_list = list(cursor)

    # 리스트를 데이터프레임으로 변환
    df = pd.DataFrame(data_list)

    # '_id' 필드를 index로 사용하지 않음
    df = df.drop('_id', axis=1)  # '_id' 컬럼 삭제
    df = df.drop('', axis=1)  # '' 컬럼 삭제

    return df

# 데이터프레임 출력
# print(df)

# CSV 파일로 저장
# df.to_csv('test_output.csv', index_label='id')

# df 이름 따로 설정 해야할 듯