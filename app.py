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

    # GPT 교정 적용
    corrected_latex = fix_latex_with_gpt(latex)
    return corrected_latex

# 문제 JSON 로딩
def load_problem_data(json_path, problem_number, subject):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return next(
            (item for item in data if item['problem_number'] == problem_number and item['subject'] == subject),
            None
        )

# ✅ 개선된 수식 비교 함수
def check_calc_error(user_latex, correct_answers):
    try:
        user_expr = latex2sympy(user_latex)
        print(f"[latex2sympy 변환 결과] {user_latex} -> {user_expr}")

        for correct in correct_answers:
            try:
                try:
                    correct_expr = latex2sympy(correct)
                except Exception:
                    correct_expr = sp.sympify(correct)

                print(f"[정답 비교] {user_expr} vs {correct_expr}")

                if sp.simplify(user_expr - correct_expr) == 0:
                    print("=> 같음 (simplify 기준)")
                    return "계산 정확함", False
                elif sp.expand(user_expr) == sp.expand(correct_expr):
                    print("=> 같음 (expand 기준)")
                    return "계산 정확함", False
                elif user_expr.equals(correct_expr):
                    print("=> 같음 (equals 기준)")
                    return "계산 정확함", False
                else:
                    print("=> 다름")

            except Exception as e:
                print(f"[정답 비교 실패] {correct} → {e}")
                continue

        return f"계산 결과가 다름. 입력식: {user_expr}", True

    except Exception as e:
        print(f"[latex2sympy 실패] {user_latex} → {e}")
        return f"수식 분석 실패: {str(e)}", True

# GPT 채점 피드백 생성기
def get_gpt_feedback(user_solution, answer, calc_errors_text):
    prompt = f"""
Student Solution:
{user_solution}

Computation Check Result:
{calc_errors_text}

Correct Answer:
{answer}

피드백 지침
1. 계산 실수는 한 줄 이상 있을 수 있어.
2. 명확한 계산 실수가 보이지 않으면 지적하지마.
3. 수학식은 무조건 LaTeX형식으로만 작성해. 
4. 실수가 없다면 짧게 "계산 실수 없어" 이렇게만 말해.
5. 문제의 의도가 풀이 방식 등은 추론하지마.
6. 오직 사용자 풀이에 적힌 수식과 결과만 보고 계산실수를 판단해.
7. 줄마다 따로따로 확인하고 실수가 있는 줄만 지적해.
8. 실수가 있는 줄이 있다면 그 줄의 수식을 보여주고 간단히 뭐가 틀렸는지 설명해.
9. "1번 줄에서, 2번 줄에서" 같은 줄 번호나 표현은 절대 쓰지 마.
10. "여기서"나 "이 줄에서" 같은 말도 쓰지 마. 
11. 계산 실수가 있을 경우만 설명해.
12. "학생 풀이:" 같은 말은 절대 쓰지 마.
13. 실수라고 생각했다가 다시 보니 맞는 경우라면 그 사실을 솔직하게 밝혀줘.
14. 친구한테 말하듯이 자연스럽고 편한 말투로 설명해.
15. 채점하기 전에 반드시 한 번 더 확인하고 답변해.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "너는 수학 선생님이야."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            top_p=0.8,
            stop=["\n\n"],
            presence_penalty=0.2,
            frequency_penalty=0.5
        )
        return response.choices[0].message.content.strip()
    except (AuthenticationError, RateLimitError, APIConnectionError) as e:
        print(f"OpenAI API 오류: {e}")
    except Exception as e:
        print(f"알 수 없는 오류: {e}")
    return None

# 📡 Flask API 엔드포인트
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

# 🖥️ 서버 실행
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
