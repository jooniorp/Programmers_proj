import re

# 파일 열기
# uploaded_file.filename은 나중에 백엔드 코드에 따라 수정
with open(uploaded_file.filename, 'r', encoding='UTF8') as file:
    # 파일의 모든 내용 읽어오기
    text = file.read()

# 텍스트에서 상대방을 찾아 opponent 변수에 할당
opponent_match = re.search(r'(.+?) 님과 카카오톡 대화', text)
opponent = opponent_match.group(1).strip() if opponent_match else None

# 상대방과 사용자 각각의 채팅이 담긴 리스트 생성
chat_matches = re.findall(r'\[([^\]]+)\] \[.*?\] (.*)', text)
user_chat = [message for sender, message in chat_matches if sender != opponent]
opponent_chat = [message for sender, message in chat_matches if sender == opponent]

# 모델 입력 데이터프레임 -> 딕셔너리 만들기
user_chat_d = {'user_chat': user_chat}
opponent_chat_d = {'opponent_chat': opponent_chat}

chat_dict = {**user_chat_d, **opponent_chat_d}
