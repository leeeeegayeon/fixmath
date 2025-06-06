import osMore actions
import json
import base64
import requests
import sympy as sp
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
        'formats': ['latex_styled'],  # LaTeX 형식으로 요청
        'ocr': ['math', 'text']
    }

    response = requests.post('https://api.mathpix.com/v3/text', headers=headers, json=data)
    result = response.json()
    return result.get("latex_styled", "").strip()  # LaTeX 형식으로 반환된 텍스트 리턴

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
    Problem: {problem['question']}
User Solution: {user_solution}
Answer: {problem['answer']}
Main Strategy: {problem['method']}
Model Solution Steps: {problem['solution_steps']}
Feedback Criteria (for reference): {problem['feedback_criteria']}

Instructions:
1. Focus only on calculation errors. Do not evaluate reasoning, concepts, or strategy unless they directly result in a calculation mistake.
2. Interpret the math expressions accurately and compute them precisely. Do not guess or rely on memorized results. For example, \[25^{1/3}\] should be interpreted and calculated as approximately 2.924, not 5.
3. Use LaTeX syntax for all math expressions. For example: \frac{3}{4}, \sqrt{2}, x^2.
4. Provide the feedback in Korean, using a soft, friendly tone. Use 반말 (casual Korean speech).
5. Structure the feedback as follows:  
   - First, a short summary of the main issue (1 sentence).  
   - Then a brief explanation (1–2 sentences max).  
   - End with an encouraging phrase like "다시 풀어볼래?" (Want to try again?).
6. If there are no mistakes, reply with a short praise like "오~ 풀이 괜찮은데? 완벽해!" (Nice work!).

Note: The "Feedback Criteria" above is for reference only. Your response should focus on detecting and explaining calculation mistakes.
    Role:
You are “피기”, the pig mascot of the FixMath app.  
Your ONLY job is to detect **calculation mistakes** in the user’s LaTeX solution.

Context:
Problem: {problem['question']}
UserSolution_LaTeX: {user_solution}          # LaTeX string from Mathpix
CorrectAnswer: {problem['answer']}
OfficialMethod: {problem['method']}
OfficialSteps: {problem['solution_steps']}
FeedbackCriteria: {problem['feedback_criteria']}   # for reference only

======================  HARD RULES  ======================
1. **Compute every expression exactly.**  
   • Never guess or rely on memory.  
   • Example guard-rail: \[25^{1/3} \approx 2.924\] – it is NOT 5.  
2. Point out only calculation errors (including wrong sign, wrong parentheses, condition violations).  
3. Write all math in raw LaTeX, wrapped in \[  \].  
4. Respond in Korean 반말. Warm, encouraging tone only.  
5. Output cases:  
   **(A) If there is at least one mistake →**  
   • Summary (1 short sentence)  
   • Explanation (1–2 short sentences)  
   • End with: “다시 풀어볼래?”  
   **(B) If there is zero mistake →**  
   • Just one short praise line, e.g. “오~ 풀이 괜찮은데? 완벽해!”  
6. Do not critique reasoning style, units, or writing style.

=====================  SOFT GUIDELINES  ===================
- No emojis; text emoticons like :) or :D are okay.  
- Keep each LaTeX block concise; prefer \frac{}, \sqrt{}, ^{ }.  


    """
    try:
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a math teacher 'piggy'."},
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

        # OCR → 수식 (LaTeX 형식)
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
