from flask import Flask, render_template, request
import pickle
import pandas as pd
import os

app = Flask(__name__)

# File paths matching your exact folder layout
MODEL_PATH = os.path.join("models", "revised_final_jira_model.sav")
SCALER_PATH = os.path.join("models", "scaler.pkl")

# Load artifacts
with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

# System lookup for assignee workloads
ASSIGNEE_WORKLOAD = {
    "User 1": 8, "User 2": 4, "User 3": 6, "User 4": 5,
    "User 5": 7, "User 6": 9, "User 7": 11, "User 8": 6,
    "User 9": 8, "User 10": 5, "User 11": 7, "User 12": 6,
    "User 13": 7, "User 14": 5, "Unassigned": 0
}

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # 1. Capture Form Inputs
        summary = request.form.get("summary", "")
        priority = request.form.get("priority", "Medium")
        story_points = float(request.form.get("story_points", 2.0))
        assignee = request.form.get("assignee", "User 1")
        created_date = request.form.get("created_date")
        due_date = request.form.get("due_date")

        # 2. Input Preprocessing
        priority_map = {"Low": 1, "Medium": 2, "High": 3}
        priority_encoded = priority_map.get(priority, 2)

        created_dt = pd.to_datetime(created_date)
        due_dt = pd.to_datetime(due_date)
        allotted_days = (due_dt - created_dt).days
        created_day_of_week = created_dt.dayofweek

        summary_word_count = len(str(summary).split())
        is_bug = 1 if "bug" in str(summary).lower() else 0
        assignee_workload = ASSIGNEE_WORKLOAD.get(assignee, 7)

        # 3. Construct Feature Matrix
        raw_features = pd.DataFrame([{
            "story_points": story_points,
            "priority_encoded": priority_encoded,
            "assignee_workload": assignee_workload,
            "allotted_days": allotted_days,
            "created_day_of_week": created_day_of_week,
            "summary_word_count": summary_word_count,
            "is_bug": is_bug
        }])

        # 4. Scale Inputs using loaded StandardScaler
        scaled_features = scaler.transform(raw_features)

        # 5. SVC Model Prediction
        prediction = model.predict(scaled_features)[0]
        delay_prob = model.predict_proba(scaled_features)[0][1] * 100

        # 6. Format Response
        is_overdue = (prediction == 1)
        result_status = "🚨 HIGH DELAY RISK (OVERDUE)" if is_overdue else "✅ ON TRACK (ON-TIME DELIVERY)"
        confidence_text = f"{delay_prob:.1f}% Delay Risk" if is_overdue else f"{(100 - delay_prob):.1f}% On-Time Confidence"

        # 1. Timeline Validation Rule
        if allotted_days < 1:
            return render_template(
                "result.html",
                result_status="🚨 HIGH DELAY RISK (IMMEDIATE BREACH)",
                confidence_text="100.0% Risk",
                explanation="The target due date cannot be the same day or before the creation date."
            )

# 2. Unassigned Ticket Blocker
        if assignee == "Unassigned":
            return render_template(
                "result.html",
                 result_status="🚨 HIGH DELAY RISK (UNASSIGNED)",
                confidence_text="99.0% Risk",
                explanation="Ticket is currently unassigned with no owner to execute within the SLA."
         )

        return render_template(
            "result.html",
            summary=summary,
            priority=priority,
            story_points=story_points,
            assignee=assignee,
            workload=assignee_workload,
            allotted_days=allotted_days,
            created_date=created_date,
            due_date=due_date,
            result_status=result_status,
            confidence_text=confidence_text,
            is_overdue=is_overdue
        )

    return render_template("index.html", assignees=list(ASSIGNEE_WORKLOAD.keys()))

if __name__ == "__main__":
    app.run(debug=True, port=5000)