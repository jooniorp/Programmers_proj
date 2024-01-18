import openai

def convert_to_mbti_style(text, mbti):
    openai.api_key = "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"
    prompt = f"'{text}'라는 문장을 {mbti} 스타일로 한글로 재작성해주세요."
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# 사용자 입력 받기
text = input("변형할 문장을 입력하세요: ")
mbti = input("MBTI 유형을 입력하세요 (예: INTP, ENFJ): ")

# 변형된 문장 출력
result = convert_to_mbti_style(text, mbti)
print("변형된 문장:", result)