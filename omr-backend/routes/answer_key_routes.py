from flask import Blueprint, request, jsonify
from models import db, AnswerKey, ScoringRule

answer_key_bp = Blueprint("answer_key", __name__)

DEFAULT_BASIC = [
    {"from": 1, "to": 40, "correct": 1, "wrong": 0, "empty": 0}
]

VALID_LEVELS = {"lower_primary", "upper_primary", "junior", "intermediate", "senior", "open"}

def _sanitize_level(raw_level):
    if not raw_level or not isinstance(raw_level, str):
        return None
    sanitized = raw_level.strip().lower().replace(" ", "_")
    if not sanitized or len(sanitized) > 64:
        return None
    return sanitized

def _parse_rule_float(rule, key, default=0.0):
    try:
        return float(rule[key])
    except (KeyError, TypeError, ValueError):
        return default

def _parse_rule_int(rule, key):
    try:
        return int(rule[key])
    except (KeyError, TypeError, ValueError):
        return None


@answer_key_bp.route("/api/save_answer_key", methods=["POST"])
def save_answer_key():
    data = request.get_json(silent=True)
    if not data or not isinstance(data, dict):
        return jsonify({"error": "Invalid or missing JSON body"}), 400

    raw_level = data.get("level", "")
    level = _sanitize_level(raw_level)
    if not level:
        return jsonify({"error": "Level required"}), 400

    answers = data.get("answers", {})
    if not isinstance(answers, dict):
        return jsonify({"error": "answers must be an object"}), 400

    scoring_rules = data.get("scoring_rules", [])
    if not isinstance(scoring_rules, list):
        return jsonify({"error": "scoring_rules must be a list"}), 400

    try:
        parsed_rules = []
        for rule in scoring_rules:
            if not isinstance(rule, dict):
                return jsonify({"error": "Each scoring rule must be an object"}), 400
            start = _parse_rule_int(rule, "from")
            end = _parse_rule_int(rule, "to")
            if start is None or end is None:
                return jsonify({"error": "Each rule must have valid 'from' and 'to' integers"}), 400
            if start > end:
                return jsonify({"error": f"Invalid range: {start} > {end}"}), 400
            parsed_rules.append({
                "from": start,
                "to": end,
                "correct": _parse_rule_float(rule, "correct", 1.0),
                "wrong": _parse_rule_float(rule, "wrong", 0.0),
                "empty": _parse_rule_float(rule, "empty", 0.0),
            })

        ranges = []
        for rule in parsed_rules:
            start, end = rule["from"], rule["to"]
            for s, e in ranges:
                if not (end < s or start > e):
                    return jsonify({
                        "error": f"Overlapping ranges: {start}-{end} conflicts with {s}-{e}"
                    }), 400
            ranges.append((start, end))

        covered = set()
        for rule in parsed_rules:
            for q in range(rule["from"], rule["to"] + 1):
                covered.add(q)

        missing = sorted([q for q in range(1, 41) if q not in covered])

        if missing:
            start = missing[0]
            prev  = missing[0]
            for i in range(1, len(missing)):
                if missing[i] == prev + 1:
                    prev = missing[i]
                else:
                    parsed_rules.append({"from": start, "to": prev, "correct": 1.0, "wrong": 0.0, "empty": 0.0})
                    start = missing[i]
                    prev  = missing[i]
            parsed_rules.append({"from": start, "to": prev, "correct": 1.0, "wrong": 0.0, "empty": 0.0})

        parsed_rules = sorted(parsed_rules, key=lambda x: x["from"])

        for q_str, ans in answers.items():
            try:
                q_no = int(str(q_str).replace("Q", "").strip())
            except (ValueError, AttributeError):
                continue
            if q_no < 1 or q_no > 40:
                continue
            ans_val = str(ans).strip().upper() if ans not in [None, ""] else ""
            existing = AnswerKey.query.filter_by(level=level, question_number=q_no).first()
            if existing:
                existing.correct_answer = ans_val
            else:
                db.session.add(AnswerKey(
                    level=level,
                    question_number=q_no,
                    correct_answer=ans_val
                ))

        if parsed_rules:
            ScoringRule.query.filter_by(level=level).delete()
            for rule in parsed_rules:
                db.session.add(ScoringRule(
                    level=level,
                    range_from=rule["from"],
                    range_to=rule["to"],
                    correct_marks=rule["correct"],
                    wrong_marks=rule["wrong"],
                    empty_marks=rule["empty"],
                ))

        db.session.commit()

        return jsonify({
            "message": "Answer key and scoring rules saved successfully",
            "note": "Missing ranges auto-filled with default scoring (1,0,0) if any"
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({"error": str(e)}), 500


@answer_key_bp.route("/api/get_answer_key/<level>")
def get_answer_key(level):
    level = _sanitize_level(level)
    if not level:
        return jsonify({"error": "Invalid level"}), 400

    try:
        keys = AnswerKey.query.filter_by(level=level).all()
        answers = {
            f"Q{str(k.question_number).zfill(2)}": k.correct_answer
            for k in keys
        }

        rules_db = ScoringRule.query.filter_by(level=level).order_by(ScoringRule.range_from).all()

        if rules_db:
            scoring_rules = [
                {
                    "from": r.range_from,
                    "to": r.range_to,
                    "correct": r.correct_marks,
                    "wrong": r.wrong_marks,
                    "empty": r.empty_marks,
                }
                for r in rules_db
            ]
        else:
            scoring_rules = DEFAULT_BASIC

        return jsonify({"answers": answers, "scoring_rules": scoring_rules})

    except Exception as e:
        return jsonify({"error": str(e)}), 500