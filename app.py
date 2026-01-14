# from flask import Flask, render_template, request, jsonify
# from chatbot import ask_bot


# import gradio as gr

# def greet(name):
#     return "Hello " + name + "!!"

# demo = gr.Interface(fn=greet, inputs="text", outputs="text")
# demo.launch()
# app = Flask(__name__)

# @app.route("/")
# def home():
#     return render_template("index.html")

# @app.route("/ask", methods=["POST"])
# def ask():
#     user_question = request.json["question"]
#     answer = ask_bot(user_question)
#     return jsonify({"answer": answer})

# if __name__ == "__main__":
#     app.run(debug=True)

from flask import Flask, render_template, request, jsonify
from chatbot import ask_bot
import os, subprocess

app = Flask(__name__)

# Auto-create embeddings if missing
if not os.path.exists("vector_store/index.faiss"):
    os.makedirs("vector_store", exist_ok=True)
    subprocess.run(["python", "embed_store.py"])

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/ask", methods=["POST"])
def ask():
    user_question = request.json["question"]
    answer = ask_bot(user_question)
    return jsonify({"answer": answer})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)

