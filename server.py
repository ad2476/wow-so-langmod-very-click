import flask
from model import ShittyClickbaitLangMod as cblm

app = flask.Flask(__name__)
m = cblm("trained_models/clickbait")
cblm.train(m)

@app.route('/')
def hello_world():
    return flask.send_file('site/main.html')
@app.route('/generate', methods=['GET'])
def generate():
    generated_clickbait = cblm.generateClickbait(m, 15)
    return generated_clickbait


@app.route('/evaluate', methods=['GET', 'POST'])
def evaluate():
    if flask.request.method == 'GET':
        phrase = flask.request.args['evaluate-text']
        print(phrase)
        eval_output = cblm.evaluateTitle(m, phrase)
        return str(eval_output)
    else:
        return "THAT ISN'T WHAT THIS ENDPOINT IS FOR"


if __name__ == '__main__':
    app.run()
