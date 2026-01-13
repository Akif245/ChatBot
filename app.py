from flask import Flask, render_template, request, jsonify
from chatbot import ask_bot

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.json["question"]
    answer = ask_bot(user_question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(debug=True)
