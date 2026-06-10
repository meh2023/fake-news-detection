from flask import Flask, render_template, request
from pathlib import Path
from src.predict import predict_news, DEFAULT_MODEL_PATH

app = Flask(__name__)

# Verify model existence
model_path = Path(DEFAULT_MODEL_PATH)
if not model_path.exists():
    print("WARNING: Model file not found. Run training script first.")

@app.route("/", methods=["GET", "POST"])
def index():
    prediction_result = None
    news_text = ""
    
    if request.method == "POST":
        action = request.form.get("action")
        if action == "clear":
            news_text = ""
        else:
            news_text = request.form.get("news_text", "")
            cleaned_text = news_text.strip()
            if cleaned_text:
                try:
                    prediction_result = predict_news(cleaned_text)
                except Exception as e:
                    prediction_result = {"error": str(e)}

    return render_template("index.html", news_text=news_text, result=prediction_result)

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
