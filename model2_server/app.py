import openai
import gradio as gr

# OpenAI API 키 설정
openai.api_key = "api키"

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
  iface.launch(share=False,server_name="0.0.0.0", server_port=2000)  # share=True로 설정하면 외부에서도 접근 가능한 링크 생성
