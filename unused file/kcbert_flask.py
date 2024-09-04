from flask import Flask, request, render_template_string
import openai

app = Flask(__name__)

# HTML 템플릿
TEMPLATE = '''
<!doctype html>
<html>
<head><title>MBTI Style Text Converter</title></head>
<body>
  <h2>MBTI Style Text Converter</h2>
  <form method="POST">
    <label for="text">Enter your text:</label><br>
    <textarea name="text" rows="4" cols="50"></textarea><br>
    <label for="mbti">Enter MBTI type (e.g., INFP, ENTJ):</label><br>
    <input type="text" name="mbti"><br><br>
    <input type="submit" value="Convert">
  </form>
  {% if converted_text %}
    <h3>Converted Text:</h3>
    <p>{{ converted_text }}</p>
  {% endif %}
</body>
</html>
'''

# OpenAI API 키 설정
openai.api_key = "your_api_key"

@app.route('/', methods=['GET', 'POST'])
def mbti_converter():
    converted_text = None
    if request.method == 'POST':
        text = request.form['text']
        mbti = request.form['mbti']
        converted_text = convert_text(text, mbti)
    return render_template_string(TEMPLATE, converted_text=converted_text)

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

if __name__ == '__main__':
    app.run(debug=True)
