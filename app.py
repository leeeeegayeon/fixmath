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
학생 풀이:
{user_solution}

계산 결과 검토:
{calc_errors_text}

문제 조건: problem["condition"]
정답: {answer}

지시
- 문제 조건·의도·해법을 절대 추측하지 마. 문제 원문은 네가 모르는거야.
- 학생 풀이에 적힌 식과 그 결과만 보고 '계산 실수 여부'를 판단해.
- 각 줄을 개별적으로 계산해 보고, 실수한 줄만 짚어 줘.
- 실수가 없다면 “계산 실수 없어” 정도로 짧게 끝내. 맞은 이유를 줄줄이 설명하지 마.
- 실수가 있으면 그 줄을 보여 주고 왜 틀렸는지만 '간단히' 설명해.
- 줄 번호(1. 2. …) 붙이지 마. 자연스럽게 문장으로 말해.
- ‘학생 풀이:’ 같은 문구는 답변에 넣지 마.
- 설명은 반말로 해.
- 수식은 LaTeX 형식으로 써.
- 답변 전에 스스로 검토해. 앞뒤 판단이 바뀌면 반드시 정정해.
- 앞부분에서 실수라고 판단했다가 뒤에서 맞다고 판단이 바뀌면 반드시 그걸 명확히 정정해.
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
