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
        'formats': ['latex_styled', 'text'],
        'ocr_options': {
            'handwriting': True  # 손글씨 인식 강화
        },
        'rm_spaces': True,         # 공백 제거
        'math_inline': False,      # 수식은 블록 형태로
        'include_latex': True,      # LaTeX 결과 명시적 포함
        'ocr': ['math', 'text']
    }

    response = requests.post('https://api.mathpix.com/v3/text', headers=headers, json=data)
    result = response.json()
    latex = result.get("latex_styled", "").strip()
    if not latex:
        latex = result.get("text", "").strip()

    return latex

# 문제 데이터 로드
def load_problem_data(json_path, problem_number, subject):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return next(
            (item for item in data if item['problem_number'] == problem_number and item['subject'] == subject),
            None
        )
#계산기
def check_calc_error(user_latex, correct_answers):
    try:
        user_expr = latex2sympy(user_latex)
        print(f"[latex2sympy 변환 결과] {user_latex} -> {user_expr}")

        for correct in correct_answers:
            try:
                correct_expr = sp.sympify(correct)
                print(f"[정답 비교] {user_expr} vs {correct_expr}")
                if sp.simplify(user_expr - correct_expr) == 0:
                    print("=> 같음 (계산 정확함)")
                    return "계산 정확함", False
                else:
                    print("=> 다름")
            except Exception as e:
                print(f"[정답 sympify 실패] {correct} → {e}")
                continue

        return f"계산 결과가 다름. 입력식: {user_expr}", True
    except Exception as e:
        print(f"[latex2sympy 실패] {user_latex} → {e}")
        return f"수식 분석 실패: {str(e)}", True

def get_gpt_feedback(user_solution, answer, calc_errors_text):
    prompt = f"""
Student Solution:
{user_solution}

Computation Check Result:
{calc_errors_text}

Correct Answer:
{answer}

Instructions
- There may be more than one calculation mistake.
- If the user's explanation is lacking, do not mention it unless it's clear that a calculation mistake occurred.
- Write all math expressions using LaTeX format only. Absolutely no exceptions.
- If there is no mistake, keep the response short, like: “계산 실수 없어”
- Never guess or mention the problem’s intent, type, or method of solving. → Assume you have never seen the original problem.
- Judge calculation mistakes only based on the expressions and results shown in the user’s solution.
- Check each line independently. Only point out lines where there is an actual mistake.
- If a mistake exists, show the expression and briefly explain what’s wrong.
- Do not add line numbers like “1.”, “2.”, etc and "첫 번째 줄에서", "두 번째 줄에서" etc. Just explain naturally and Only comment on lines with actual miscalculations.
- Do not include phrases like “Student Solution:” in your reply.
- Use casual, informal language (like talking to a friend).
- Before answering, double-check your judgment.
- If you first say it’s wrong but later realize it’s right, clearly say so.
- Reply in Korean only. Do not answer in English or any other language.
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
