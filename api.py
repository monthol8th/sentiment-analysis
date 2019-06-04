from flask import Flask, jsonify, request
from pythainlp.tokenize import word_tokenize
import re
import emoji
import sklearn
import pickle
from identity_tokenizer import identity_tokenizer


with open('vectorize.pkl', 'rb') as vectorize_file:
    tfidf = pickle.load(vectorize_file)

with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

emoji_spliter = emoji.get_emoji_regexp().split


def split_emoji(x): return [a for y in x for a in emoji_spliter(
    y) if not bool(re.search(r'^\s*$', a))]


def remove_dupes(x): return [re.sub(r'^([ก-ฮ])\1{3,}$', r'\1\1', t) for t in x]


def replace_555(x): return [re.sub(r'^5{3,}$', r'LAUGHT', t) for t in x]


sentiment = {-1: "negative", 0: "neutral", 1: "positive"}

app = Flask(__name__)


@app.errorhandler(404)
def page_not_found(e):
    # note that we set the 404 status explicitly
    return jsonify(
        success="false",
        error="not found"
    ), 404


@app.errorhandler(500)
def page_not_found(e):
    # note that we set the 404 status explicitly
    return jsonify(
        success="false",
        error="internal server error"
    ), 500


@app.route("/")
def index():
    return jsonify(
        success="true"
    ), 200


@app.route("/api")
def api():
    return jsonify(
        success="true"
    ), 200


@app.route("/api/analyze")
def analyze():
    text = request.args.get('text')
    token = word_tokenize(text)
    token = split_emoji(token)
    token = remove_dupes(token)
    token = replace_555(token)

    vector = tfidf.transform([token]).toarray()

    predict = model.predict(vector)[0]

    return jsonify(
        success="true",
        sentiment=sentiment[predict]
    ), 200


if __name__ == "__main__":
    app.run()
