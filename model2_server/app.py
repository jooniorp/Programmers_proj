from flask import Flask, request, render_template_string, jsonify
import openai
from pymongo import MongoClient
import gradio as gr

# OpenAI API 키 설정
openai.api_key = "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"

def convert_text(text, mbti):
    # 여기에 MBTI 스타일로 변환하는 로직을 구현합니다.
    # 예: "Please rewrite this in [MBTI] style: [text]"
    prompt = f"'{text}'라는 문장을 {mbti} 스타일로 한글로 재작성해주세요."
    response = openai.Completion.create(
        engine="gpt-3.5-turbo-instruct",
        prompt=prompt,
        max_tokens=150
    )
    return response.choices[0].text.strip()

# Gradio 인터페이스 생성
iface = gr.Interface(
    fn=convert_text,
    inputs=[gr.Textbox(), gr.Dropdown(["INTJ", "INTP", "ENTJ", "ENTP", "INFJ", "INFP", "ENFJ", "ENFP", "ISTJ", "ISFJ", "ESTJ", "ESFJ", "ISTP", "ISFP", "ESTP", "ESFP"], label="MBTI")],
    outputs=gr.Textbox()
)

# Gradio 웹 애플리케이션 실행
if __name__ == '__main__':
    # Gradio 웹 애플리케이션 실행
  iface.launch(share=True,server_name="0.0.0.0", server_port=2000)  # share=True로 설정하면 외부에서도 접근 가능한 링크 생성
'''
# MongoDB 연결 설정
client = MongoClient('mongodb:27017')  # 'mongodb'는 MongoDB 컨테이너의 이름
db = client['testdb']  # 데이터베이스 이름
collection = db['testcollection']  # 컬렉션 이름

# 텍스트 데이터를 받아 MongoDB에 저장
@app.route('/save_to_mongo', methods=['POST'])
def save_to_mongo():
    data = request.get_json()
    text_data = data.get('text_data', '')

     # MongoDB에 데이터 저장
    result = collection.insert_one({'text_data': text_data})

    return jsonify({'result': 'Data saved to MongoDB!', 'mongo_id': str(result.inserted_id)})

'''
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
