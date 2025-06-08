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


def get_gpt_feedback(user_solution, answer):
    messages = [
        {"role": "system", "content": "너는 수학 선생님이야. 학생 풀이를 보고 계산 실수를 알려줘."},
        {"role": "user", "content": f"학생 풀이:\n{user_solution}"},
        {"role": "user", "content": f"계산 결과 검토 결과:\n{calc_errors_text}"},
        {"role": "user", "content": f"정답: {answer}"},
        {"role": "user", "content": "계산 실수가 있는 줄이 있다면 그 줄만 지적해줘. 중간 과정은 추측하지 말고, 반말로 간단히 설명해줘. 수식은 LaTeX으로 써줘."}
    ]
    try:
        response = client.chat.completions.create(
            model="gpt-4-turbo",
            messages=messages,
            temperature=1,
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

        feedback = get_gpt_feedback(user_solution, problem["answer"], errors_txt)
        if not feedback:
            return jsonify({"error": "GPT 피드백 실패"}), 500

        return jsonify({
            "user_solution": user_solution,
            "calc_errors": calc_errors,
            "feedback": feedback
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
