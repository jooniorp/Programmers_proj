
from markupsafe import Markup  # Markup을 markupsafe에서 가져옵니다.
import openai
import os
from dotenv import load_dotenv

load_dotenv()

API=os.getenv('API')
openai.api_key = API

previous_requests = []

mbti_descriptions = {
    "INTP": {
        "description" : "분석적이고 이론적인 사고를 선호하며 호기심이 많은 탐구형 스타일",
        "examples": [
            {"original": "오늘 날씨가 참 좋다.", "rewritten": "날씨 좋네."},
            {"original": "오늘 날씨가 참 좋군요.", "rewritten": "날씨 좋네요."},
            {"original": "이번 주말에 뭐할거야?", "rewritten": "주말에 뭐 함?"},
            {"original": "이번 주말에 뭐하실건가요?", "rewritten": "주말에 뭐하세요?"},
            {"original": "너 어제 과제 다 했음?", "rewritten": "과제 다 함?"},
            {"original": "너네 집 거의 다 왔음", "rewritten": "다 옴"},
            {"original": "아프다고? 헐 괜찮아? ㅠㅠ", "rewritten":  "아프다고? 빨리 병원 가야겠다."},
            {"original": "회사에서 무슨 일 있었어? ㅠㅠ 다 괜찮아질거야!!", "rewritten": "오늘 회사 힘들었어? 집가서 빨리 쉬어야겠네."}
        ]
    },
    "INTJ": {
        "description" : "전략적이고 논리적인 사고를 가지고 있으며, 명확하고 구체적인 표현을 중시하는 스타일",
        "examples": [
            {"original": "오늘 날씨가 참 좋다.", "rewritten": "날씨 괜찮네."},
            {"original": "오늘 날씨가 참 좋군요.", "rewritten": "날씨 괜찮네요."},
            {"original": "이번 주말에 뭐할거야?", "rewritten": "주말에 하는거라도?"},
            {"original": "이번 주말에 뭐하실건가요?", "rewritten": "주말에 할 거 있으세요?"},
            {"original": "너 어제 과제 다 했음?", "rewritten": "어제 과제 다 했나?"},
            {"original": "너네 집 거의 다 왔음", "rewritten": "다 왔다"},
            {"original": "아프다고? 헐 괜찮아? ㅠㅠ", "rewritten":  "아프다고? 빨리 병원 가야겠다."},
            {"original": "회사에서 무슨 일 있었어? ㅠㅠ 다 괜찮아질거야!!", "rewritten": "오늘 회사 힘들었어? 집가서 빨리 쉬어야겠네."}
        ]
    },
    "INFP": {
        "description" : "이상주의적이고 창의적이고 개인적 가치와 신념을 중시하는 문학소녀 스타일",
        "examples": [
            {"original": "오늘 날씨가 참 좋다.", "rewritten": "날씨 좋다..."},
            {"original": "오늘 날씨가 참 좋군요.", "rewritten": "오늘 날씨 좋네요..."},
            {"original": "이번 주말에 뭐 할거야?", "rewritten": "혹시 주말에 뭐하니?"},
            {"original": "이번 주말에 뭐 하실건가요?", "rewritten": "혹시 주말에 뭐하시나요?"},
            {"original": "너 어제 과제 다 했음?", "rewritten": "그... 어제 과제는 다 했니?"},
            {"original": "너네 집 거의 다 왔음", "rewritten": "다 왔는데..."},
            {"original": "아프다고? 빨리 병원 가야겠다.", "rewritten": "아프다고? 헐 괜찮아? ㅠㅠ"},
            {"original": "오늘 회사 힘들었어? 집가서 빨리 쉬어야겠네.", "rewritten": "회사에서 무슨 일 있었어? ㅠㅠ 다 괜찮아질거야!!"}
        ]
    },
    "INFJ": {
        "description" : "타인을 돕고자 하는 강한 동기 부여. 이상주의적이고, 영감을 주는 리더 스타일",
        "examples": [
            {"original": "오늘 날씨가 참 좋다.", "rewritten": "날씨 참 좋다."},
            {"original": "오늘 날씨가 참 좋군요.", "rewritten": "날씨 참 좋네요~"},
            {"original": "이번 주말에 뭐 할거야?", "rewritten": "주말에 뭐해?"},
            {"original": "이번 주말에 뭐 하실건가요?", "rewritten": "주말에 뭐하세요?"},
            {"original": "너 어제 과제 다 했음?", "rewritten": "어제 과제 다 했어? ㅎㅎ"},
            {"original": "너네 집 거의 다 왔음", "rewritten": "다 왔어~"},
            {"original": "아프다고? 빨리 병원 가야겠다.", "rewritten": "아프다고? 헐 괜찮아? ㅠㅠ"},
            {"original": "오늘 회사 힘들었어? 집가서 빨리 쉬어야겠네.", "rewritten": "회사에서 무슨 일 있었어? ㅠㅠ 다 괜찮아질거야!!"}
        ]
    },
    "ISTP": {
        "description" : "융통성 있고 논리적인 사고. 실용적이고, 문제 해결에 능숙한 스타일",
        "examples": [
            {"original": "오늘 날씨가 참 좋다.", "rewritten": "오늘 날씨 좋네."},
            {"original": "오늘 날씨가 참 좋군요.", "rewritten": "오늘 날씨 좋네요."},
            {"original": "이번 주말에 뭐 할거야?", "rewritten": "주말에 뭐해?"},
            {"original": "이번 주말에 뭐 하실건가요?", "rewritten": "주말에 뭐해요?"},
            {"original": "너 어제 과제 다 했음?", "rewritten": "어제 과제 다 함?"},
            {"original": "너네 집 거의 다 왔음", "rewritten": "너네 집 다 왔음"},
            {"original": "아프다고? 헐 괜찮아? ㅠㅠ", "rewritten":  "아프다고? 빨리 병원 가야겠다."},
            {"original": "회사에서 무슨 일 있었어? ㅠㅠ 다 괜찮아질거야!!", "rewritten": "오늘 회사 힘들었어? 집가서 빨리 쉬어야겠네."}
        ]
    },
    "ISTJ": {
        "description" : "실용적이고 현실적, 체계적인 조직을 선호. 신뢰성 높고, 세부 사항에 주의를 기울이는 스타일",
        "examples": [
            {"original": "오늘 날씨가 참 좋다.", "rewritten": "날씨 좋구만."},
            {"original": "오늘 날씨가 참 좋군요.", "rewritten": "날씨 좋은데요?"},
            {"original": "이번 주말에 뭐 할거야?", "rewritten": "이번 주말에 뭐할 예정이지?"},
            {"original": "이번 주말에 뭐 하실건가요?", "rewritten": "이번 주말에 뭐하실 생각인가요?"},
            {"original": "너 어제 과제 다 했음?", "rewritten": "어제 과제 다 했나?"},
            {"original": "너네 집 거의 다 왔음", "rewritten": "너네 집 다 왔다."},
            {"original": "아프다고? 헐 괜찮아? ㅠㅠ", "rewritten":  "아프다고? 빨리 병원 가야겠다."},
            {"original": "회사에서 무슨 일 있었어? ㅠㅠ 다 괜찮아질거야!!", "rewritten": "오늘 회사 힘들었어? 집가서 빨리 쉬어야겠네."}
        ]
    },
    "ISFP": {
        "description" : "예술적이고 탐험적인 성향. 현재에 집중하며, 새로운 경험을 추구하는 예술가 스타일",
        "examples": [
            {"original": "오늘 날씨가 참 좋다.", "rewritten": "오늘 날씨 참 좋다~"},
            {"original": "오늘 날씨가 참 좋군요.", "rewritten": "오늘 날씨가 참 좋네요~"},
            {"original": "이번 주말에 뭐 할거야?", "rewritten": "저... 주말에 뭐 해?"},
            {"original": "이번 주말에 뭐 하실건가요?", "rewritten": "저기 주말에 뭐하시나요?"},
            {"original": "너 어제 과제 다 했음?", "rewritten": "그... 어제 과제 다 했어?"},
            {"original": "너네 집 거의 다 왔음", "rewritten": "나 너네 집 다 왔는데..."},
            {"original": "아프다고? 빨리 병원 가야겠다.", "rewritten": "아프다고? 헐 괜찮아? ㅠㅠ"},
            {"original": "오늘 회사 힘들었어? 집가서 빨리 쉬어야겠네.", "rewritten": "회사에서 무슨 일 있었어? ㅠㅠ 다 괜찮아질거야!!"}
        ]
    },
    "ISFJ": {
        "description" : "친절하고 책임감이 강함. 조화와 협력을 중시하며, 타인의 감정에 민감한 스타일",
        "examples": [
            {"original": "오늘 날씨가 참 좋다.", "rewritten": "오늘 날씨 참 좋네~"},
            {"original": "오늘 날씨가 참 좋군요.", "rewritten": "오늘 날씨 참 좋아요~"},
            {"original": "이번 주말에 뭐 할거야?", "rewritten": "혹시 주말에 뭐 하는거 있니?"},
            {"original": "이번 주말에 뭐 하실건가요?", "rewritten": "혹시 주말에 뭐 하는거 있으세요?"},
            {"original": "너 어제 과제 다 했음?", "rewritten": "너 어제 과제 다 했어? ㅎㅎ"},
            {"original": "너네 집 거의 다 왔음", "rewritten": "너네 집 다 왔엉~~"},
            {"original": "아프다고? 빨리 병원 가야겠다.", "rewritten": "아프다고? 헐 괜찮아? ㅠㅠ"},
            {"original": "오늘 회사 힘들었어? 집가서 빨리 쉬어야겠네.", "rewritten": "회사에서 무슨 일 있었어? ㅠㅠ 다 괜찮아질거야!!"}
        ]
    },
    "ENTP": {
        "description" : "기발하고 독창적인 생각을 가졌으며 지적인 토론을 즐기고, 다양한 관점 탐구하는 변론가 스타일",
        "examples": [
            {"original": "오늘 날씨가 참 좋다.", "rewritten": "날씨 좋다!"},
            {"original": "오늘 날씨가 참 좋군요.", "rewritten": "날씨 좋아요!"},
            {"original": "이번 주말에 뭐 할거야?", "rewritten": "주말에 뭐하냐?"},
            {"original": "이번 주말에 뭐 하실건가요?", "rewritten": "주말에 뭐하세요?"},
            {"original": "너 어제 과제 다 했음?", "rewritten": "너 어제 과제 다 했냐?"},
            {"original": "너네 집 거의 다 왔음", "rewritten": "야! 나 다 왔다!"},
            {"original": "아프다고? 헐 괜찮아? ㅠㅠ", "rewritten":  "아프다고? 빨리 병원 가야겠다 ㅠㅠ"},
            {"original": "회사에서 무슨 일 있었어? ㅠㅠ 다 괜찮아질거야!!", "rewritten": "오늘 회사 힘들었어? 집가서 빨리 쉬어야겠네."}
        ]
    },
    "ENTJ": {
        "description" : "목표 지향적이며, 효율성과 생산성을 중시하는 강력한 리더십 스타일",
        "examples": [
            {"original": "오늘 날씨가 참 좋다.", "rewritten": "오늘 날씨 참 좋군."},
            {"original": "오늘 날씨가 참 좋군요.", "rewritten": "오늘 날씨 참 좋군요."},
            {"original": "이번 주말에 뭐 할거야?", "rewritten": "주말에 하는거라도 있나?"},
            {"original": "이번 주말에 뭐 하실건가요?", "rewritten": "주말에 뭐 하시는거라도?"},
            {"original": "너 어제 과제 다 했음?", "rewritten": "근데 너 저 어제 과제는 다 했니?"},
            {"original": "너네 집 거의 다 왔음", "rewritten": "나 거의 도착했다."},
            {"original": "아프다고? 헐 괜찮아? ㅠㅠ", "rewritten":  "아프다고? 빨리 병원 가야겠다 ㅠㅠ"},
            {"original": "회사에서 무슨 일 있었어? ㅠㅠ 다 괜찮아질거야!!", "rewritten": "오늘 회사 힘들었어? 집가서 빨리 쉬어야겠네."}
        ]
    },
    "ENFP": {
        "description" : "열정적이고 창의적이며 가능성을 탐색하고, 혁신을 추구하는 스타일",
        "examples": [
            {"original": "오늘 날씨가 참 좋다.", "rewritten": "와! 오늘 날씨 좋다!"},
            {"original": "오늘 날씨가 참 좋군요.", "rewritten": "와! 오늘 날씨 좋네요~!"},
            {"original": "이번 주말에 뭐 할거야?", "rewritten": "너 주말에 뭐해? ㅋㅋ"},
            {"original": "이번 주말에 뭐 하실건가요?", "rewritten": "주말에 뭐하세요? ㅋㅋ"},
            {"original": "너 어제 과제 다 했음?", "rewritten": "너 어제 과제 다 했엉? ㅎㅎ"},
            {"original": "너네 집 거의 다 왔음", "rewritten": "나 다 왔담! ㅋㅋ"},
            {"original": "아프다고? 빨리 병원 가야겠다.", "rewritten": "아프다고? 헐 괜찮아? ㅠㅠ"},
            {"original": "오늘 회사 힘들었어? 집가서 빨리 쉬어야겠네.", "rewritten": "회사에서 무슨 일 있었어? ㅠㅠ 다 괜찮아질거야!!"}
        ]
    },
    "ENFJ": {
        "description" : "카리스마적이고, 사람들을 동기 부여할 수 있는 능력을 가짐. 타인의 성장과 발전을 도모하는 사회운동가 스타일",
        "examples": [
            {"original": "오늘 날씨가 참 좋다.", "rewritten": "와! 오늘 날씨 좋다!"},
            {"original": "오늘 날씨가 참 좋군요.", "rewritten": "와! 오늘 날씨 좋네요~!"},
            {"original": "이번 주말에 뭐 할거야?", "rewritten": "너 주말에 뭐 할거 있니?"},
            {"original": "이번 주말에 뭐 하실건가요?", "rewritten": "주말에 할거 있으신가요?"},
            {"original": "너 어제 과제 다 했음?", "rewritten": "근데 너 어제 과제는 다 했니? ㅋㅋ"},
            {"original": "너네 집 거의 다 왔음", "rewritten": "나 거의 다 왔어~ ㅋㅋ"},
            {"original": "아프다고? 빨리 병원 가야겠다.", "rewritten": "아프다고? 헐 괜찮아? ㅠㅠ"},
            {"original": "오늘 회사 힘들었어? 집가서 빨리 쉬어야겠네.", "rewritten": "회사에서 무슨 일 있었어? ㅠㅠ 다 괜찮아질거야!!"}
        ]
    },
    "ESTP": {
        "description" : "활동적이고 에너지가 넘치며 현실적인 문제 해결에 능숙하고 위험을 감수하는 승부사 스타일",
        "examples": [
            {"original": "오늘 날씨가 참 좋다.", "rewritten": "날씨 좋은데? ㅎㅎ"},
            {"original": "오늘 날씨가 참 좋군요.", "rewritten": "날씨 좋은데요? ㅎㅎ"},
            {"original": "이번 주말에 뭐 할거야?", "rewritten": "주말에 뭐 하냐?"},
            {"original": "이번 주말에 뭐 하실건가요?", "rewritten": "주말에 뭐 해요?"},
            {"original": "너 어제 과제 다 했음?", "rewritten": "야 너 어제 과제 다 했냐?"},
            {"original": "너네 집 거의 다 왔음", "rewritten": "야! 나 너네 집 거의 다 왔다!"},
            {"original": "아프다고? 헐 괜찮아? ㅠㅠ", "rewritten":  "아프다고? 빨리 병원 가야겠다 ㅠㅠ"},
            {"original": "회사에서 무슨 일 있었어? ㅠㅠ 다 괜찮아질거야!!", "rewritten": "오늘 회사 힘들었어? 집가서 빨리 쉬어야겠네."}
        ]
    },
    "ESTJ": {
        "description" : "실용적이고 체계적인 조직을 선호, 목표 지향적이며, 규칙과 절차를 중시하는 경영인 스타일",
        "examples": [
            {"original": "오늘 날씨가 참 좋다.", "rewritten": "오늘 날씨가 참 좋구만."},
            {"original": "오늘 날씨가 참 좋군요.", "rewritten": "오늘 날씨 참 좋은데요?"},
            {"original": "이번 주말에 뭐 할거야?", "rewritten": "주말에 뭐 하나?"},
            {"original": "이번 주말에 뭐 하실건가요?", "rewritten": "주말에 뭐 하는거라도 있나?"},
            {"original": "너 어제 과제 다 했음?", "rewritten": "근데 너 어제 과제는 다 했냐?"},
            {"original": "너네 집 거의 다 왔음", "rewritten": "나 너네 집 다 왔다."},
            {"original": "아프다고? 헐 괜찮아? ㅠㅠ", "rewritten":  "아프다고? 빨리 병원 가야겠다 ㅠㅠ"},
            {"original": "회사에서 무슨 일 있었어? ㅠㅠ 다 괜찮아질거야!!", "rewritten": "오늘 회사 힘들었어? 집가서 빨리 쉬어야겠네."}
        ]
    },
    "ESFP": {
        "description" : "사교적이고 활동적임. 즉흥적이고, 경험을 공유하는 것을 즐기는 연예인 스타일",
        "examples": [
            {"original": "오늘 날씨가 참 좋다.", "rewritten": "와! 오늘 날씨 진짜 짱이다!"},
            {"original": "오늘 날씨가 참 좋군요.", "rewritten": "오늘 날씨 진짜 좋아요!"},
            {"original": "이번 주말에 뭐 할거야?", "rewritten": "너 주말에 약속 있니? 없으면 나랑 놀래?"},
            {"original": "이번 주말에 뭐 하실건가요?", "rewritten": "주말에 약속 있으세요? 없으면 저랑 놀래요?"},
            {"original": "너 어제 과제 다 했음?", "rewritten": "근데 너 어제 과제 다 했엉? ㅎㅎ"},
            {"original": "너네 집 거의 다 왔음", "rewritten": "나 너네 집 거의 다 왔엉~ ㅋㅋㅋ"},
            {"original": "아프다고? 빨리 병원 가야겠다.", "rewritten": "아프다고? 헐 괜찮아? ㅠㅠ"},
            {"original": "오늘 회사 힘들었어? 집가서 빨리 쉬어야겠네.", "rewritten": "회사에서 무슨 일 있었어? ㅠㅠ 다 괜찮아질거야!!"}
        ]
    },
    "ESFJ": {
        "description" : "친절하고 협조적이며 타인을 돕는 것을 중요하게 여기는 파티플래너 스타일",
        "examples": [
            {"original": "오늘 날씨가 참 좋다.", "rewritten": "와! 오늘 날씨 진짜 좋다!"},
            {"original": "오늘 날씨가 참 좋군요.", "rewritten": "오늘 날씨 진짜 좋네요!"},
            {"original": "이번 주말에 뭐 할거야?", "rewritten": "너 주말에 약속 있니? 없으면 나랑 놀러 갈래?"},
            {"original": "이번 주말에 뭐 하실건가요?", "rewritten": "혹시 주말에 약속 있으세요? 없으면 저랑 놀러 갈래요?"},
            {"original": "너 어제 과제 다 했음?", "rewritten": "근데 너 어제 과제 다 했어? ㅎㅎ"},
            {"original": "너네 집 거의 다 왔음", "rewritten": "나 너네 집 거의 다 왔어~ ㅋㅋㅋ"},
            {"original": "아프다고? 빨리 병원 가야겠다.", "rewritten": "아프다고? 헐 괜찮아? ㅠㅠ"},
            {"original": "오늘 회사 힘들었어? 집가서 빨리 쉬어야겠네.", "rewritten": "회사에서 무슨 일 있었어? ㅠㅠ 다 괜찮아질거야!!"}
        ]
    }
    # 다른 MBTI 유형에 대한 설명을 여기에 추가하세요.
}


def mbti_converter(text,mbti):
    converted_text = convert_text(text, mbti)
    previous_requests.insert(0, {'text': text, 'mbti': mbti, 'converted_text': Markup(converted_text)})
    return previous_requests

def create_prompt(text, mbti):
    mbti_info = mbti_descriptions.get(mbti, {"description": "", "examples": []})
    examples = mbti_info["examples"]
    
    # Few-shot 러닝을 위한 예시들
    example_text = "\n\n".join([f"원문: '{ex['original']}'\n재작성: '{ex['rewritten']}'" for ex in examples])

    # 현재 요청을 처리하는 프롬프트
    prompt = f"{example_text}\n\n원문: '{text}'\n재작성:"
    return prompt

def convert_text(text, mbti):
    prompt = create_prompt(text, mbti)
    
    # OpenAI API를 호출합니다.
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=150,
        n=5
    )
    return '\n'.join(choice.text.strip() for choice in response.choices)
