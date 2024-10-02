from flask import Flask, render_template, request
import pickle

app = Flask(__name__)
# load the model
model = pickle.load(open('mlmodel.sav', 'rb'))

@app.route('/')
def home():
    result = ''
    return render_template('index.html', **locals())

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    gross_income = int(request.form['gross_income'])
    total_cost = int(request.form['total_cost'])
    quantity = int(request.form['quantity'])
    result = model.predict([[gross_income, total_cost, quantity]])[0]
    return render_template('index.html', **locals())

if __name__ == '__main__':
    app.run(debug=True)