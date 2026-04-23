from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class StudentOMR(db.Model):
    __tablename__ = "student_omr_results"

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100))
    level = db.Column(db.String(20))
    centre_number = db.Column(db.String(20))
    dob = db.Column(db.String(20))
    answers = db.Column(db.JSON)
    score = db.Column(db.Integer)


class AnswerKey(db.Model):
    __tablename__ = "answer_key"

    id = db.Column(db.Integer, primary_key=True)
    level = db.Column(db.String(20))
    question_number = db.Column(db.Integer)
    correct_answer = db.Column(db.String(1))