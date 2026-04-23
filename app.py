from flask import Flask, request, jsonify, render_template
from config import Config
from models import db, StudentOMR, AnswerKey
import os
from omr_processor import process_omr_file
import pymysql

pymysql.install_as_MySQLdb()

app = Flask(__name__)
app.config.from_object(Config)

db.init_app(app)

# Create DB
with app.app_context():
    db.create_all()


@app.route("/")
def home():
    return render_template("index.html")
# -------------------------------
# API: PROCESS OMR
# -------------------------------
@app.route("/api/upload", methods=["POST"])
def upload():
    if "file" not in request.files:
        return jsonify({"error": "No file uploaded"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    filepath = os.path.join(app.config["UPLOAD_FOLDER"], file.filename)
    file.save(filepath)

    result = process_omr_file(filepath)

    if "error" in result:
        return jsonify(result), 500

    name = result["name"]
    centre = result["centre_number"]
    answers = result["answers"]
    # level = result["level"] 
    level = "intermediate"

    keys = AnswerKey.query.filter_by(level=level).all()
    key_dict = {k.question_number: k.correct_answer for k in keys}

    score = 0

    for q, ans in answers.items():
        q_num = int(q)
        correct = key_dict.get(q_num)

        if ans is None:
            continue

        if 1 <= q_num <= 25:
            if ans == correct:
                score += 2
            else:
                score += 0

        elif 26 <= q_num <= 35:
            if ans == correct:
                score += 3
            else:
                score -= 1

        elif 36 <= q_num <= 40:
            if ans == correct:
                score += 4
            else:
                score -= 2

    student = StudentOMR(
        name=name,
        level=level,
        centre_number=centre,
        dob=None,
        answers=answers,
        score=score
    )

    db.session.add(student)
    db.session.commit()

    return jsonify({
        "name": name,
        "centre_number": centre,
        "level": level,
        "score": score,
        "answers": answers
    })


@app.route("/api/results", methods=["GET"])
def get_results():
    students = StudentOMR.query.all()

    data = []
    for s in students:
        data.append({
            "id": s.id,
            "name": s.name,
            "centre_number": s.centre_number,
            "level": s.level,
            "score": s.score,
            "answers": s.answers
        })

    return jsonify(data)


@app.route("/api/seed-answer-key", methods=["POST"])
def seed_answer_key():
    data = request.get_json()

    level = data.get("level")
    answers = data.get("answers")

    if not level or not answers:
        return jsonify({"error": "Invalid data"}), 400

    AnswerKey.query.filter_by(level=level).delete()

    for q, ans in answers.items():
        record = AnswerKey(
            level=level,
            question_number=int(q),
            correct_answer=ans
        )
        db.session.add(record)

    db.session.commit()

    return jsonify({"message": "Answer key seeded successfully"})


if __name__ == "__main__":
    app.run(debug=True)