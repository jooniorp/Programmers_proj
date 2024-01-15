from pymongo import MongoClient
from mobile_text_to_dict import mobile_text_to_dict_from_file

def save_to_mongodb(my_dict):
    # MongoDB에 연결
    client = MongoClient("mongodb://localhost:27017/")
    db = client["test_database"]
    collection = db["test_collection"]

    # MongoDB에 저장하기 위해 변환
    mongo_document = {}
    for key, value in my_dict.items():
        mongo_document[key] = value

    # MongoDB에 문서 삽입
    result = collection.insert_one(mongo_document)

    # 결과 확인
    print(f"Inserted document ID: {result.inserted_id}")

if __name__ == "__main__":
    import sys

    if len(sys.argv) != 2:
        print("Usage: python save_to_mongodb.py <file_path>")
        sys.exit(1)

    file_path = sys.argv[1]
    my_dict = mobile_text_to_dict_from_file(file_path)

    # MongoDB에 저장
    save_to_mongodb(my_dict)


"""
이제 각 파일은 명령줄에서 실행할 때 텍스트 파일의 경로를 매개변수로 전달받습니다.

예를 들어:
in bash)

python read_and_create_dict.py your_text_file.txt
python save_to_mongodb.py your_text_file.txt

이렇게 하면 다른 이름의 텍스트 파일을 사용하여 각각의 스크립트를 실행할 수 있습니다.
"""