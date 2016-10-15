import flask
from backend.model import ShittyClickbaitLangMod as cblm

app = flask.Flask(__name__)
m = cblm("backend/trained_models")
cblm.train(m)

@app.route('/')
def hello_world():
    return flask.send_file('main.html')
@app.route('/generate', methods=['GET'])
def generate():
    generated_clickbait = cblm.generateClickbait(m, 15)
    return generated_clickbait


@app.route('/evaluate', methods=['GET', 'POST'])
def evaluate():
    if flask.request.method == 'GET':
        phrase = flask.request.form['evaluate-text']
        eval_output = cblm.evaluateTitle(m, phrase)
        return eval_output
    else:
        return "THAT ISN'T WHAT THIS ENDPOINT IS FOR"


if __name__ == '__main__':
    app.run()