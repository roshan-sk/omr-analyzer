from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class StudentOMR(db.Model):
    __tablename__ = "student_omr_results"

    id = db.Column(db.Integer, primary_key=True)
    file_name = db.Column(db.String(255))
    name = db.Column(db.String(100))
    level = db.Column(db.String(20))
    centre_number = db.Column(db.String(20))
    dob = db.Column(db.String(20))
    score = db.Column(db.Integer)
    answers = db.Column(db.JSON)
    verify_ans = db.Column(db.JSON)
    batch_id = db.Column(db.String(50))


class AnswerKey(db.Model):
    __tablename__ = "answer_key"

    id = db.Column(db.Integer, primary_key=True)
    level = db.Column(db.String(20))
    question_number = db.Column(db.Integer)
    correct_answer = db.Column(db.String(1))