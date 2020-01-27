from flask import Flask, request, jsonify
import reg_predict
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def run():
    data = request.get_json(force=True)
    input_params = data['input']
    #result =  predict.predict(input_params)
    result =  predict_reg(input_params)
    return jsonify({'prediction': result})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7775)