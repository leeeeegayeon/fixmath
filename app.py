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

def define_symbols_from_latex(latex_text):
    tokens = re.findall(r"[a-zA-Z_][a-zA-Z_0-9]*", latex_text)
    reserved = {"frac", "sqrt", "sum", "int", "lim", "log", "sin", "cos", "tan", "dx", "dy", "dt", "d", "ln", "pi", "infty", "left", "right"}
    symbols_needed = sorted(set(tokens) - reserved)
    symbol_dict = {name: sp.Symbol(name) for name in symbols_needed}
    globals().update(symbol_dict)
    return symbol_dict

def line_has_calc_error(latex_line):
    if '=' not in latex_line:
        return False, None
    try:
        left_raw, right_raw = latex_line.split('=', 1)
        left_val = sp.N(parse_latex(left_raw))
        right_val = sp.N(parse_latex(right_raw))
        if abs(left_val - right_val) > 1e-6:
            return True, f"{latex_line.strip()}  \u27A1  {left_val} \u2260 {right_val}"
        else:
            return False, None
    except Exception as e:
        return True, f"\u26A0\ufe0f 파싱 실패: {latex_line.strip()}  ({e})"

def detect_calc_errors(latex_text):
    error_lines = []
    lines = latex_text.replace('\\\n', '\n').replace('\\', '\n').splitlines()
    for raw in lines:
        line = raw.strip()
        if not line:
            continue
        is_err, msg = line_has_calc_error(line)
        if is_err:
            error_lines.append(msg or line)
    return error_lines

def get_gpt_feedback(user_solution, answer, calc_errors_text):
    prompt = f"""
학생 풀이: {user_solution}
계산 결과 검토 결과:
{calc_errors_text}

정답: {answer}

- 위의 학생 풀이와 자동 계산 결과를 참고해서, 계산 실수가 있는지 판단해줘.
- 중간 과정은 추측하지 말고, 주어진 줄과 결과만 가지고 설명해줘.
- 실수한 줄이 있다면 왜 틀렸는지 간단히 설명해줘.
- 설명은 반말로 해줘.
- 피드백을 할때 수식은 LaTex수식으로 변환해줘.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "너는 수학 선생님이야."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.2,
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
        define_symbols_from_latex(user_solution)
        calc_errors = detect_calc_errors(user_solution)
        errors_txt = "\n".join(calc_errors) if calc_errors else "없음"

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
