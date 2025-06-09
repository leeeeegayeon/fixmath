import os
import json
import base64
import requests
import sympy as sp
from latex2sympy2 import latex2sympy
from flask import Flask, request, jsonify
from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError
from werkzeug.utils import secure_filename
from dotenv import load_dotenv

# .env 불러오기
load_dotenv()
MATHPIX_APP_ID = os.getenv('MATHPIX_APP_ID')
MATHPIX_APP_KEY = os.getenv('MATHPIX_APP_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Mathpix OCR
def mathpix_ocr(image_path):
    with open(image_path, "rb") as image_file:
        image_base64 = base64.b64encode(image_file.read()).decode()

    headers = {
        'app_id': MATHPIX_APP_ID,
        'app_key': MATHPIX_APP_KEY,
        'Content-type': 'application/json'
    }

    data = {
        'src': f'data:image/png;base64,{image_base64}',
        'formats': ['latex_styled'],
        'ocr': ['math', 'text']
    }

    response = requests.post('https://api.mathpix.com/v3/text', headers=headers, json=data)
    result = response.json()
    return result.get("latex_styled", "").strip()

# 문제 데이터 로드
def load_problem_data(json_path, problem_number, subject):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return next(
            (item for item in data if item['problem_number'] == problem_number and item['subject'] == subject),
            None
        )

# 계산기: latex 수식 → sympy 수식 비교
def check_calc_error(user_latex, correct_answers):
    try:
        user_expr = latex2sympy(user_latex)
        for correct in correct_answers:
            try:
                correct_expr = sp.sympify(correct)
                if sp.simplify(user_expr - correct_expr) == 0:
                    return "계산 정확함", False
            except:
                continue
        return f"계산 결과가 다름. 입력식: {user_expr}", True
    except Exception as e:
        return f"수식 분석 실패: {str(e)}", True

def get_gpt_feedback(user_solution, answer, calc_errors_text):
    prompt = f"""
{user_solution}
계산 결과 검토 결과:
{calc_errors_text}

정답: {answer}

- 문제 조건, 함수 정의, 문제 유형, 문제 해법, 의도 등을 유추하지 마.
- 문제 원문을 보지 않았다는 점을 전제로, 학생 풀이와 결과만 가지고 판단해.
- 학생 풀이를 참고해서 '계산' 실수가 있는지 판단해줘.
- 줄이 이전 줄에서 이어진 것처럼 보여도, 계산은 줄마다 따로 다시 확인해.
- 중간 과정은 추측하지 말고, 주어진 줄과 '결과'만 가지고 설명해줘.
- '계산 실수'한 줄이 있다면 왜 틀렸는지 간단히 설명해줘.
- 설명은 반말로 해줘.
- 피드백을 할 때 수식은 LaTeX 수식으로 변환해줘.
- 답변하기 전에 스스로 검토를 거쳐.
- 정답이면 다시 풀어보라는 말은 하지 마.
- 앞부분에서 실수라고 판단했다가 뒤에서 맞다고 판단이 바뀌면 반드시 그걸 명확히 정정해.
- '학생 풀이:', '계산 결과 검토 결과:' 같은 문장은 GPT가 다시 설명하지 마.
- 줄 번호(예: 1. ~~, 2. ~~)를 붙이지 말고, 자연스럽게 설명해.
- 계산 실수가 없다면 간단하게 말해줘. 각 줄이 맞은 이유를 하나하나 다시 설명하지 마.
- 실수가 있는 경우에 해당 줄을 골라서 설명해줘.

"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "너는 수학 선생님이야."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    except (AuthenticationError, RateLimitError, APIConnectionError) as e:
        print(f"OpenAI API 오류: {e}")
    except Exception as e:
        print(f"알 수 없는 오류: {e}")
    return None

@app.route("/analyze", methods=["POST"])
def analyze():
    try:
        image_file = request.files.get("file")
        filename = request.form.get("filename")

        if not image_file or not filename:
            return jsonify({"error": "파일 또는 파일명이 없습니다."}), 400

        safe_name = secure_filename(filename)
        save_path = os.path.join(UPLOAD_FOLDER, safe_name)
        image_file.save(save_path)

        parts = safe_name.replace(".png", "").split("_")
        if len(parts) != 4:
            return jsonify({"error": "파일명 형식 오류"}), 400

        json_path = f"{parts[0]}_{parts[1]}.json"
        subject = parts[2]
        problem_number = int(parts[3])

        user_solution = mathpix_ocr(save_path)

        problem = load_problem_data(json_path, problem_number, subject)
        if not problem or problem["subject"] != subject:
            return jsonify({"error": f'{problem_number}, {json_path}, {problem["subject"]}, {subject}'}), 404

        calc_errors_text, has_error = check_calc_error(user_solution, problem["answer"])

        feedback = get_gpt_feedback(user_solution, problem["answer"], calc_errors_text)
        if not feedback:
            return jsonify({"error": "GPT 피드백 실패"}), 500

        return jsonify({
            "user_solution": user_solution,
            "calc_errors": calc_errors_text,
            "feedback": feedback
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
