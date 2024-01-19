import re

def mobile_text_to_dict_from_file(file_path, user_mbti, opponent_mbti):
    # file_path는 나중에 백엔드 코드에 따라 수정해야될 수 있음
    with open(file_path, 'r', encoding='UTF8') as file:
        # 파일의 모든 내용 읽어오기
        text = file.read()

    # 텍스트에서 상대방을 찾아 opponent 변수에 할당
    opponent_match = re.search(r'(.+?) 님과 카카오톡 대화', text)
    opponent = opponent_match.group(1).strip() if opponent_match else None

    # 대화를 추출하는 정규 표현식
    mobile_pattern = r'[^\n]*\n\d{4}년 \d{1,2}월 \d{1,2}일 [^\n]*,\s([^\n]+) : (.*)'

    # 정규 표현식을 사용하여 대화 추출
    matches = re.findall(mobile_pattern, text)

    # URL 형식 패턴
    url_pattern = re.compile(r'https?://\S+')
    # app 공유 패턴
    app_sharing_pattern = re.compile(r'\[.+?\].+')

    # jpg 파일 형식 패턴
    jpg_pattern = re.compile(r'\b\w+\.jpg\b')
    # mp4 파일 형식 패턴
    mp4_pattern = re.compile(r'\b\w+\.mp4\b')
    # m4a 파일 형식 패턴
    m4a_pattern = re.compile(r'\b\w+\.m4a\b')
    # vcf 파일 형식 패턴
    vcf_pattern = re.compile(r'\b\w+\.vcf\b')

    # 입금 완료 패턴
    receive_pattern = re.compile(r'(\d+)원 받기 완료!')
    # 송금 패턴
    send_pattern = re.compile(r'(\d+)원을 보냈어요.')
    # 음악 공유 패턴
    music_pattern = re.compile(r"'(.+)' 음악을 공유했습니다.")

    # 파일 업로드 패턴
    file_pattern = re.compile(r'파일: (.+\.\w+)')
    # 지도 공유 패턴
    map_address_pattern = re.compile(r'지도: (.+)')


    # 상대방과 사용자 각각의 채팅이 담긴 리스트 생성
    user_chat = []
    opponent_chat = []
    

    for sender, message in matches:
        if url_pattern.search(message): # 링크가 포함된 경우 메시지를 무시하고 다음으로 진행
            continue
        elif app_sharing_pattern.search(message): # app 공유 무시
            continue
        elif message == '이모티콘': # 이모티콘 무시
            continue
        elif jpg_pattern.search(message): # 사진 공유 무시
            continue
        elif mp4_pattern.search(message): # 동영상 공유 무시
            continue
        elif m4a_pattern.search(message): # 음성파일 공유 무시
            continue
        elif vcf_pattern.search(message): # 연락처 공유 무시
            continue
        elif receive_pattern.search(message): # 입금 무시
            continue
        elif send_pattern.search(message): # 송금 무시
            continue
        elif music_pattern.search(message): # 음악 공유 무시
            continue
        elif file_pattern.search(message): # 파일 업로드 무시
            continue
        elif map_address_pattern.search(message): # 주소 공유 무시
            continue
        elif message == '카카오톡 프로필': # 프로필 공유 무시
            continue
        
        # 상대방과 사용자 각각의 채팅이 담긴 리스트 생성
        if sender != opponent:
            user_chat.append(message)
        elif sender == opponent:
            opponent_chat.append(message)


    # 모델 입력 데이터프레임 -> 딕셔너리 만들기
    user_chat_d = {'user_chat': user_chat, 'user_mbti' : user_mbti}
    opponent_chat_d = {'opponent_chat': opponent_chat, 'opponent_mbti' : opponent_mbti}

    return user_chat_d, opponent_chat_d



def pc_text_to_dict_from_file(file_path, user_mbti, opponent_mbti):
    # file_path는 나중에 백엔드 코드에 따라 수정해야될 수 있음
    with open(file_path, 'r', encoding='UTF8') as file:
        # 파일의 모든 내용 읽어오기
        text = file.read()

    # 텍스트에서 상대방을 찾아 opponent 변수에 할당
    opponent_match = re.search(r'(.+?) 님과 카카오톡 대화', text)
    opponent = opponent_match.group(1).strip() if opponent_match else None

    # 대화를 추출하는 정규 표현식
    pc_pattern = r'\[([^\]]+)\] \[.*?\] (.*)'

    # 정규 표현식을 사용하여 대화 추출
    matches = re.findall(pc_pattern, text)


    # URL 형식 패턴
    url_pattern = re.compile(r'https?://\S+')
    # app 공유 패턴
    app_sharing_pattern = re.compile(r'\[.+?\].+')

    # 입금 완료 패턴
    receive_pattern = re.compile(r'(\d+)원 받기 완료!')
    # 송금 패턴
    send_pattern = re.compile(r'(\d+)원을 보냈어요.')
    # 음악 공유 패턴
    music_pattern = re.compile(r"'(.+)' 음악을 공유했습니다.")

    # 파일 업로드 패턴
    file_pattern = re.compile(r'파일: (.+\.\w+)')
    # 지도 공유 패턴
    map_address_pattern = re.compile(r'지도: (.+)')

    # 연락처 공유 패턴
    share_contactinfo_pattern = re.compile(r'연락처: .+')

    # 상대방과 사용자 각각의 채팅이 담긴 리스트 생성
    user_chat = []
    opponent_chat = []
    temp_chat = ''
    flag = 'user' if matches[0] != opponent else 'opponent'

    for sender, message in matches:
        if url_pattern.search(message): # 링크가 포함된 경우 메시지를 무시하고 다음으로 진행
            continue
        elif app_sharing_pattern.search(message): # app 공유 무시
            continue
        elif message == '이모티콘': # 이모티콘 무시
            continue
        elif message == '사진': # 사진 공유 무시
            continue
        elif message == '동영상':  # 동영상 공유 무시
            continue
        elif message == '음성메시지':  # 음성파일 공유 무시
            continue
        elif receive_pattern.search(message): # 입금 무시
            continue
        elif send_pattern.search(message): # 송금 무시
            continue
        elif music_pattern.search(message): # 음악 공유 무시
            continue
        elif file_pattern.search(message): # 파일 업로드 무시
            continue
        elif map_address_pattern.search(message): # 주소 공유 무시
            continue
        elif message == '카카오톡 프로필': # 프로필 공유 무시
            continue
        elif share_contactinfo_pattern.search(message):
            continue
        
        # 상대방과 사용자 각각의 채팅이 담긴 리스트 생성
        if sender != opponent and flag == 'opponent':
            user_chat.append(temp_chat)
            flag = 'user'
        elif sender == opponent and flag == 'user':
            opponent_chat.append(temp_chat)
            flag = 'opponent'

        temp_chat += message



    # 모델 입력 데이터프레임 -> 딕셔너리 만들기
    user_chat_d = {'user_chat': user_chat, 'user_mbti' : user_mbti}
    opponent_chat_d = {'opponent_chat': opponent_chat, 'opponent_mbti' : opponent_mbti}

    return user_chat, opponent_chat



def process_text_file(file_path, user_mbti, opponent_mbti):
    # file_path는 나중에 백엔드 코드에 따라 수정해야될 수 있음
    with open(file_path, 'r', encoding='UTF8') as file:
        # 파일의 모든 내용 읽어오기
        text = file.read()

    mobile_pattern = r'[^\n]*\n\d{4}년 \d{1,2}월 \d{1,2}일 [^\n]*,\s([^\n]+) : (.*)'
    pc_pattern = r'\[([^\]]+)\] \[.*?\] (.*)'

    # 특정 패턴을 찾아 모바일 채팅 데이터와 PC 채팅 데이터를 구별
    if re.search(mobile_pattern, text):
        return mobile_text_to_dict_from_file(file_path, user_mbti, opponent_mbti)
    elif re.search(pc_pattern, text):
        return pc_text_to_dict_from_file(file_path, user_mbti, opponent_mbti)
    else:
        print("Unrecognized file format")
