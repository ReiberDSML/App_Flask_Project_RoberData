



from flask import Flask, request, render_template
from pickle import load

app = Flask(__name__)

model = load(open('../models/naive_bayes_gaussian_opt.sav', 'rb'))

class_dict = {
    0: 'Negative Comment',
    1: 'Positive Comment'
}

@app.route('/', methods=['GET', 'POST'])
def index():
    pred_comment = None  # Inicializar predicción
    if request.method == 'POST':
        text = request.form['text']
        if text:  # Verificar si se ingresó texto
            # Asegurarse de que el texto se pase como lista al modelo
            prediction = model.predict([text])
            pred_comment = class_dict[prediction[0]]  # Obtener el resultado correcto

    return render_template('index.html', prediction=pred_comment)

if __name__ == '__main__':
    app.run(debug=True)