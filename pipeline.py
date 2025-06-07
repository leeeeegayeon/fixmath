import os
import json
import base64
import requests
import sympy as sp
import re
from flask import Flask, request, jsonify
from openai import OpenAI, AuthenticationError, RateLimitError, APIConnectionError
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
from sympy.parsing.latex import parse_latex

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

# 변수 자동 정의
def define_symbols_from_latex(latex_text):
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z_0-9]*", latex_text)
    reserved = {"frac", "sqrt", "sum", "int", "lim", "log", "sin", "cos", "tan", "dx", "dy", "dt", "d", "ln", "pi", "infty", "left", "right"}
    symbols_needed = sorted(set(tokens) - reserved)
    symbol_dict = {name: sp.Symbol(name) for name in symbols_needed}
    globals().update(symbol_dict)
    return symbol_dict

# 학생 풀이 최종 결과 계산
def evaluate_expression(latex_text):
    try:
        expr = parse_latex(latex_text)
        result = expr.evalf()
        return float(result)
    except Exception as e:
        return None

# GPT 피드백 생성
def get_gpt_feedback(user_result, correct_answer, is_correct):
    prompt = f"""
학생의 계산 결과는 {user_result} 이고, 정답은 {correct_answer} 입니다.
결과를 비교하면 {'맞았습니다' if is_correct else '틀렸습니다'}.

- 풀이과정은 추측하지 마세요.
- 단순 계산 실수 또는 숫자 오타인지 간단하게 판단해서 짧고 친절하게 피드백을 작성해 주세요.
- 불필요하게 장황한 설명은 피해주세요.
"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "너는 수학 선생님이야. 학생에게 친절하고 짧게 피드백을 해줘."},
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

# Flask API endpoint
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

        # OCR 및 파싱
        user_latex = mathpix_ocr(save_path)
        define_symbols_from_latex(user_latex)
        user_result = evaluate_expression(user_latex)

        if user_result is None:
            return jsonify({"error": "학생 풀이 수식 파싱 실패"}), 500

        problem = load_problem_data(json_path, problem_number, subject)
        if not problem or problem["subject"] != subject:
            return jsonify({"error": f'{problem_number}, {json_path}, {problem["subject"]}, {subject}'}), 404

        correct_answer = float(problem["answer"])
        is_correct = abs(user_result - correct_answer) < 1e-6

        feedback = get_gpt_feedback(user_result, correct_answer, is_correct)
        if not feedback:
            return jsonify({"error": "GPT 피드백 실패"}), 500

        return jsonify({
            "user_solution": user_latex,
            "user_result": user_result,
            "is_correct": is_correct,
            "feedback": feedback
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 서버 시작
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
