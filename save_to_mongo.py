from pymongo import MongoClient
from textfile_to_dict import mobile_text_to_dict_from_file, pc_text_to_dict_from_file, process_text_file # 프로젝트 구조에 따라 수정해야할 수 있음

def save_to_mongodb(chat_dict_user, chat_dict_opponent):
    # MongoDB에 연결
    # 인증이 필요할 때, pymongo.MongoClient("mongodb://사용자이름:비밀번호@localhost:27017/")
    client = MongoClient("mongodb://ies:6b5@localhost:27017/")
    # 어떤 database, collection에 저장할지 정해야함.
    db = client["input_database"]
    user_chat_collection = db["user_chat"]
    opponent_chat_collection = db["opponent_chat"]

    # MongoDB에 저장하기 위해 변환
    mongo_document = {}
    for key, value in my_dict.items():
        mongo_document[key] = value

    # 컬렉션에 데이터 삽입
    user_chat_collection.insert_one(chat_dict_user)
    opponent_chat_collection.insert_one(chat_dict_opponent)




if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2 or len(sys.argv) > 4:
        print("Usage: python textfile_save_to_mongodb.py <file_path> [<user_mbti> [<opponent_mbti>]]")
        sys.exit(1)

    file_path = sys.argv[1]
    
    mbti_list = ['ISTJ', 'ISFJ', 'INFJ', 'INTJ', 'ISTP', 'ISFP', 'INFP', 'INTP', 'ESTP', 'ESFP', 'ENFP', 'ENTP', 'ESTJ', 'ESFJ', 'ENFJ', 'ENTJ']

    user_mbti = sys.argv[2] if sys.argv[2] in mbti_list else None
    opponent_mbti = sys.argv[3] if sys.argv[3] in mbti_list else None

    # 텍스트 데이터 처리 함수 실행
    my_dict = process_text_file(file_path, user_mbti, opponent_mbti)

    # user_chat과 opponent_chat 딕셔너리 분리
    chat_dict_user = {'user_chat': my_dict.get('user_chat', []), 'user_mbti': my_dict.get('user_mbti')}
    chat_dict_opponent = {'opponent_chat': my_dict.get('opponent_chat', []), 'opponent_mbti': my_dict.get('opponent_mbti')}

    # MongoDB에 저장
    save_to_mongodb(chat_dict_user, chat_dict_opponent)


    # in bash
    # 입력 값이 없는 경우 아무 값이나 입력
    # python textfile_save_to_mongodb.py your_text_file.txt user_mbti opponent_mbti
    
# 웹에서 업로드된 파일과 웹에서 입력 받는 user_mbti, opponent_mbti에 대한 코드 작성은 웹 제작이
# 진행되고 작성할 수 있을 듯.