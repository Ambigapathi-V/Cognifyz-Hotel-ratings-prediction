from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

# Mock prediction function
def predict_aggregate_rating(price, location, service):
    # Example logic for prediction
    return round((price * 0.4 + location * 0.3 + service * 0.3), 2)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.form
    price = float(data.get('price'))
    location = float(data.get('location'))
    service = float(data.get('service'))

    # Call the prediction function
    prediction = predict_aggregate_rating(price, location, service)

    return jsonify({'prediction': prediction})

if __name__ == '__main__':
    app.run(debug=True)
