from flask import Flask, render_template, request
from utils import check_misinformation

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    results = []
    if request.method == 'POST':
        text = request.form['article_text']
        results = check_misinformation(text)
    return render_template('index.html', results=results)

if __name__ == '__main__':
    app.run(debug=True)

