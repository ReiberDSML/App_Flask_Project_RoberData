



from flask import Flask, request, render_template
from pickle import load
import numpy as np

app = Flask(__name__)

# Cargar modelo y vectorizador
model = load(open('../models/naive_bayes_gaussian_opt.sav', 'rb'))
vectorizer = load(open('../models/tfidf_vectorizer.sav', 'rb'))

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
            # Preprocesar el texto usando el vectorizador cargado
            text_vectorized = vectorizer.transform([text])
            # Convertir a formato denso para GaussianNB
            text_dense = text_vectorized.toarray()
            # Realizar la predicción
            prediction = model.predict(text_dense)
            pred_comment = class_dict[prediction[0]]  # Obtener el resultado correcto

    return render_template('index.html', prediction=pred_comment)

import os
# Parte final modificada para que funcione en Render
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))  # Usar el puerto asignado o 5000 por defecto
    app.run(host='0.0.0.0', port=port)
