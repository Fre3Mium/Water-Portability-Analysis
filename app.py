from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('classifier.pkl', 'rb'))

app = Flask(__name__)



@app.route('/')
def man():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def home():
    data1 = request.form['ph']
    data2 = request.form['hardness']
    data3 = request.form['solids']
    data4 = request.form['chloramines']
    data5 = request.form['sulfate']
    data6 = request.form['conductivity']
    data7 = request.form['organic_carbon']
    data8 = request.form['trihalomethanes']
    data9 = request.form['turbidity']

    arr = np.array([[data1, data2, data3, data4, data5, data6, data7, data8, data9]])
    pred = model.predict(arr)
    return render_template('results1.html', data=pred)


if __name__ == "__main__":
    app.run(debug=True)