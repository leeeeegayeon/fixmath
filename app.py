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
- 불필요한 재해석 없이 최대한 원래 풀이의 의도를 유지해 주세요.
- 수식 흐름을 유지하세요.
- 불필요한 수식 재구성이나 문맥과 어긋나는 새로운 구조 생성은 하지 마세요.
- 주어진 수식 내의 풀이 흐름 내에서만 추론하여 오타를 고쳐주세요.
- 중간에 생략된 풀이과정이 있더라도 임의로 추가하지 마세요.
- 수식을 재구성하거나 새롭게 해석하지 마세요.
- 변수 이름 'b, d, l, o'와 숫자 '6, 0, 1'와 같이 혼동될 수 있는 문자들은 혼동하지 말고 문맥으로 구분해서 고쳐주세요.
- 오타 교정만 하고 계산 오류가 있더라도 계산실수 검산은 하지 마세요.
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
    except Exception as e:
        print(f"[GPT 오류] {e}")
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

    corrected_latex = fix_latex_with_gpt(latex)
    return corrected_latex

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
    matches = re.findall(r"\\*([a-zA-Z])\s*=\s*(-?\d+(?:\.\d+)?)", latex_text)
    return {sp.Symbol(var): sp.sympify(val) for var, val in matches}

# 사용된 변수 추출
def extract_all_used_variables(lines):
    symbols = set()
    for line in lines:
        try:
            expr = latex2sympy(line)
            symbols |= expr.free_symbols
        except:
            continue
    return symbols

# ✅ 통합 개선된 계산 오류 체크 함수
def check_calc_error(user_latex, correct_answers, problem_known_vars=None):
    try:
        lines = [line.strip() for line in user_latex.split('\\\\') if line.strip()]

        defined_vars = {}
        for line in reversed(lines):
            defs = extract_variable_definitions_from_latex(line)
            if defs:
                defined_vars.update(defs)

        user_used_vars = extract_all_used_variables(lines)
        if problem_known_vars:
            known_syms = set(sp.Symbol(k) for k in problem_known_vars)
            used_forbidden_vars = user_used_vars & known_syms
            if used_forbidden_vars:
                return f"문제에 이미 사용된 변수 {', '.join(str(v) for v in used_forbidden_vars)}를 임의로 사용했습니다.", True

        for idx, line in enumerate(lines):
            if '=' in line:
                try:
                    lhs_str, rhs_str = line.split('=')
                    lhs = latex2sympy(lhs_str.strip()).subs(defined_vars)
                    rhs = latex2sympy(rhs_str.strip()).subs(defined_vars)
                    if sp.simplify(lhs - rhs) != 0:
                        return f"{idx+1}번째 줄 계산 결과 다름: {lhs} ≠ {rhs}", True
                except Exception as e:
                    return f"{idx+1}번째 줄 수식 해석 실패: {e}", True

        # 마지막 줄 비교
        if lines:
            try:
                user_final = lines[-1].split('=')[0].strip()
                user_expr = latex2sympy(user_final).subs(defined_vars)
                for correct in correct_answers:
                    try:
                        try:
                            correct_expr = latex2sympy(correct).subs(defined_vars)
                        except Exception:
                            correct_expr = sp.sympify(correct).subs(defined_vars)

                        if sp.simplify(user_expr - correct_expr) == 0:
                            return "계산 정확함", False
                        elif sp.expand(user_expr) == sp.expand(correct_expr):
                            return "계산 정확함", False
                        elif user_expr.equals(correct_expr):
                            return "계산 정확함", False
                    except:
                        continue
                return "최종 결과가 정답과 다름", True
            except Exception as e:
                return f"최종 수식 분석 실패: {e}", True
        else:
            return "수식이 비어 있음", True

    except Exception as e:
        return f"전체 수식 분석 실패: {e}", True

# GPT 피드백 생성
def get_gpt_feedback(user_solution, answer, calc_errors_text):
    prompt = f"""
    - 사용자 풀이에 적힌 수식과 결과만 보고 계산실수를 판단해.
학생 풀이:
{user_solution}

계산이 오류 난 식:
{calc_errors_text}

정답:
{answer}

- 계산 실수는 한 줄 이상 있을 수 있어.
- 수학식은 무조건 LaTeX형식으로만 작성해. 
- 실수가 없다면 짧게 '계산 실수 없어' 이렇게만 말해.
- 문제의 의도나 풀이 방식 등은 추론하지마.
- 실수가 있는 줄이 있다면 간단히 뭐가 틀렸는지 설명해.
- '학생 풀이:' 같은 말은 절대 쓰지 마.
- 친절한 말투로 설명해.
- ' x = 5에서 계산 실수 있어. 2x = 9면 x는 4.5야.' 처럼 답변해줘.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "너는 수학 선생님이야."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            top_p=0.8
        )
        return response.choices[0].message.content.strip()
    except (AuthenticationError, RateLimitError, APIConnectionError) as e:
        print(f"OpenAI API 오류: {e}")
    except Exception as e:
        print(f"알 수 없는 오류: {e}")
    return None

# 📡 Flask API
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

# 🖥️ 서버 실행
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)


