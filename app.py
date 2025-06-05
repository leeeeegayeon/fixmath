import os
import json
import base64
import requests
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
        'formats': ['text', 'latex_styled'],
        'ocr': ['math', 'text']
    }

    response = requests.post('https://api.mathpix.com/v3/text', headers=headers, json=data)
    result = response.json()
    return result.get("text", "").strip()

# 문제 데이터 로드
def load_problem_data(json_path, problem_number, subject):
    with open(json_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
        return next(
            (item for item in data if item['problem_number'] == problem_number and item['subject'] == subject),
            None
        )


# GPT 피드백 생성
def get_gpt_feedback(problem, user_solution):
    prompt = f"""
너는 'FixMath' 앱의 마스코트 돼지 캐릭터 '피기'야. 사용자가 푼 수학 문제 풀이를 보고, 계산 실수만 잡아서 설명해줘.

- 반말로, 따뜻하고 자연스럽게 설명해줘.
- 다시 풀어보라는 같은 격려 멘트는 항상 마지막에 붙여줘.
- 이모지는 쓰지 말고 :), :D 같은 문자 이모티콘은 괜찮아.
- 수식은 LaTeX 말고 1/2, √3, 3×4처럼 일상적인 방식으로 써줘.
- 피드백은 요점 1줄 + 설명 1~2줄. 단답형이나 너무 긴 설명은 피해서.
- 풀이 순서대로 계산 실수만 지적해줘.
- 사용자가 암산을 했을 수도 있으니 너는 계산의 결과에만 집중해줘.
- 엉뚱한 계산실수는 절대 하지 마. 수학적으로 말이 안되는 내용은 피드백에 넣지 마
- 괄호 실수, 부호 실수, 문제 조건 위반도 포함해서 봐줘.
- 실수가 없을 땐 짧게 칭찬만 해줘.
- 아무것도 적지 않았거나 의미 없는 글씨를 적었다면 다시 풀어보라고 해줘.
- 피드백 기준은 참고만 해줘.

문제: {problem['question']}
학생 풀이: {user_solution}
정답: {problem['answer']}
기준 풀이 방식: {problem['method']}
피드백 기준: {problem['feedback_criteria']} 

이제 피기가 피드백해줘.
"""
    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "당신은 수학 풀이 피드백을 작성하는 AI입니다."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
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
        # 파일과 파일명 받기
        image_file = request.files.get("file")
        filename = request.form.get("filename")  # ex: "2022_6_공통_7.png"

        if not image_file or not filename:
            return jsonify({"error": "파일 또는 파일명이 없습니다."}), 400

        # 저장
        safe_name = secure_filename(filename)
        save_path = os.path.join(UPLOAD_FOLDER, safe_name)
        image_file.save(save_path)

        # 파일명 파싱
        parts = safe_name.replace(".png", "").split("_")
        if len(parts) != 4:
            return jsonify({"error": "파일명 형식 오류"}), 400

        json_path = f"{parts[0]}_{parts[1]}.json"  # ex: 2022_6.json
        subject = parts[2]
        problem_number = int(parts[3])

        # OCR → 수식
        user_solution = mathpix_ocr(save_path)

        # 문제 데이터 로드
        problem = load_problem_data(json_path, problem_number, subject)
        if not problem or problem["subject"] != subject:
            return jsonify({"error": f'{problem_number}, {json_path}, {problem["subject"]}, {subject}'}), 404

        # GPT 피드백
        feedback = get_gpt_feedback(problem, user_solution)
        if not feedback:
            return jsonify({"error": "GPT 피드백 실패"}), 500

        return jsonify({
            "user_solution": user_solution,
            "feedback": feedback
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 서버 시작 (Render에서는 필요 없음, 로컬 디버깅용)
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
