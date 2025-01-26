



from flask import Flask, request, render_template
from pickle import load

app = Flask(__name__)

model = load(open('..\models\naive_bayes_gaussian_opt.sav', 'rb'))

class_dict = {
    0: 'Negative Comment',
    1: 'Positive Comment'
}

@app.route('/', methods=['GET', 'POST'])

def index():
    if request.method == 'POST':

        text = request.form['text']
        prediction = model.predict(text)
        pred_comment = class_dict (prediction)

    else:
        pred = None

    return render_template('index.html', prediction=pred_comment)

    