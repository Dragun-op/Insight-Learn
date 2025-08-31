from flask import Blueprint, render_template, request, flash, current_app
from flask_login import login_required, current_user
from app.models import db, Explanation
from app.nlp.understanding_classifier import classify_explanation
from app.forms import ClassifierForm
import requests

nlp = Blueprint("nlp", __name__)

HF_QG_MODEL = "mrm8488/t5-base-finetuned-question-generation-ap"

@nlp.route("/", endpoint="index")
def index():
    return render_template("nlp/index.html")

@nlp.route("/classifier", methods=["GET", "POST"])
@login_required
def classifier():
    form = ClassifierForm()

    if form.validate_on_submit():
        topic = form.topic.data
        explanation = form.explanation.data
        hf_api_key = current_app.config["HF_API_TOKEN"]
        headers = {"Authorization": f"Bearer {hf_api_key}"}

        if topic and not explanation:
            payload = {
                "inputs": f"context: {topic} answer: {topic.split()[0]}",
                "options": {"use_cache": False}
            }

            q_resp = requests.post(
                f"https://api-inference.huggingface.co/models/{HF_QG_MODEL}",
                headers=headers,
                json=payload
            )

            if q_resp.status_code != 200:
                print("QG request failed:", q_resp.status_code, q_resp.text)
                flash("Could not generate question", "danger")
                return render_template("nlp/classifier.html", form=form)

            q_text = q_resp.json()[0]["generated_text"].strip()
            return render_template("nlp/classifier.html", form=form, topic=topic, question=q_text)

        elif explanation and topic:
            result = classify_explanation(explanation, hf_api_key)

            exp = Explanation(
                text=explanation,
                result=result["prediction"],
                score_understood=result["scores"].get("Understood", 0.0),
                score_memorized=result["scores"].get("Memorized", 0.0),
                score_confused=result["scores"].get("Confused", 0.0),
                user_id=current_user.id
            )

            db.session.add(exp)
            db.session.commit()

            return render_template("nlp/result.html",
                                   input_text=explanation,
                                   scores=result["scores"].items(),
                                   result=result["prediction"])

    return render_template("nlp/classifier.html", form=form)
