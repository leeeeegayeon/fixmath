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
import re

# .env 불러오기
load_dotenv()
MATHPIX_APP_ID = os.getenv('MATHPIX_APP_ID')
MATHPIX_APP_KEY = os.getenv('MATHPIX_APP_KEY')
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')

client = OpenAI(api_key=OPENAI_API_KEY)

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# 🔧 GPT로 LaTeX 수식 교정
def fix_latex_with_gpt(raw_latex):
    prompt = f"""
다음은 Mathpix에서 인식한 LaTeX 수식입니다. 문법 오류, 괄호 짝, 연산자 누락, 흐름상 부자연스러운 표현 등이 있을 수 있습니다. 자연스럽고 정확한 수식으로 고쳐주세요.

[LaTeX 입력]
{raw_latex}

[지시사항]
- 문법적으로 유효한 LaTeX로 수정해 주세요.
- 설명하지 말고, 수정된 LaTeX만 한 줄로 출력하세요.
- 최대한 원래 풀이의 의도를 유지해 주세요.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "너는 수학 LaTeX 오류를 고치는 도구야."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
        )
        return response.choices[0].message.content.strip()
    except Exception:
        return raw_latex

# 🧠 Mathpix OCR + GPT 교정 포함
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
        'formats': ['latex_styled', 'text'],
        'ocr_options': {'handwriting': True},
        'rm_spaces': True,
        'math_inline': False,
        'include_latex': True,
        'ocr': ['math', 'text']
    }

    response = requests.post('https://api.mathpix.com/v3/text', headers=headers, json=data)
    result = response.json()

    latex = result.get("latex_styled", "").strip()
    if not latex:
        latex = result.get("text", "").strip()

    return fix_latex_with_gpt(latex)

# JSON 문제 로드
def load_problem_data(json_path, problem_number, subject):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return next(
            (item for item in data if item['problem_number'] == problem_number and item['subject'] == subject),
            None
        )

# 변수 정의 추출
def extract_variable_definitions_from_latex(latex_text):
    matches = re.findall(r"\\*([a-zA-Z])\\s*=\\s*(-?\\d+(?:\\.\\d+)?)", latex_text)
    return {sp.Symbol(var): sp.sympify(val) for var, val in matches}

def extract_all_used_variables(lines):
    symbols = set()
    for line in lines:
        try:
            expr = latex2sympy(line)
            symbols |= expr.free_symbols
        except:
            continue
    return symbols

# 계산 실수 검출
def check_calc_error(user_latex, correct_answers, problem_known_vars=None):
    try:
        lines = [line.strip() for line in user_latex.split('\\\') if line.strip()]
        defined_vars = {}
        for line in reversed(lines):
            defs = extract_variable_definitions_from_latex(line)
            if defs:
                defined_vars.update(defs)

        used_vars = extract_all_used_variables(lines)
        if problem_known_vars:
            known_syms = set(sp.Symbol(k) for k in problem_known_vars)
            if used_vars & known_syms:
                return "이미 주어진 변수를 임의로 사용했음", True

        for idx, line in enumerate(lines):
            if '=' not in line:
                continue
            try:
                lhs, rhs = line.split('=')
                lhs_val = latex2sympy(lhs).subs(defined_vars)
                rhs_val = latex2sympy(rhs).subs(defined_vars)
                if sp.simplify(lhs_val - rhs_val) != 0:
                    return f"{idx+1}번째 줄 계산 오류: {lhs_val} ≠ {rhs_val}", True
            except Exception as e:
                return f"{idx+1}번째 줄 해석 실패: {e}", True

        if lines:
            try:
                final_lhs = lines[-1].split('=')[0].strip()
                user_expr = latex2sympy(final_lhs).subs(defined_vars)
                for correct in correct_answers:
                    try:
                        try:
                            correct_expr = latex2sympy(correct).subs(defined_vars)
                        except:
                            correct_expr = sp.sympify(correct).subs(defined_vars)

                        if sp.simplify(user_expr - correct_expr) == 0 or user_expr.equals(correct_expr):
                            return "계산 정확함", False
                    except:
                        continue
                return "최종 결과가 정답과 다름", True
            except Exception as e:
                return f"최종 수식 해석 실패: {e}", True
        return "수식 없음", True
    except Exception as e:
        return f"전체 분석 실패: {e}", True

# GPT 피드백 생성
def get_gpt_feedback(user_solution, answer, calc_errors_text):
    prompt = f"""
학생 풀이:
{user_solution}

계산이 오류 난 식:
{calc_errors_text}

정답:
{answer}

[피드백 지침]
- 수학식은 LaTeX 형식만 써
- 실수한 줄만 짚고 간단히 왜 틀렸는지 말해
- 실수 없으면 '계산 실수 없어' 한 줄만
- 설명은 친구에게 말하듯 자연스럽게
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "너는 수학 선생님이야."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
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

        if not problem:
            return jsonify({"error": f'{problem_number}, {json_path} 에서 문제를 찾을 수 없습니다'}), 404

        if problem["subject"] != subject:
            return jsonify({"error": f'과목 불일치: {problem["subject"]} vs {subject}'}), 404

        calc_errors_text, has_error = check_calc_error(user_solution, problem["answer"], problem.get("known_vars"))
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

@app.get("/")
def read_root():
    return {"message": "Hello World"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
