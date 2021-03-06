from flask import Flask, request, jsonify
#import predict
#from predict2 import predict_class as predict
app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def run():
    data = request.get_json(force=True)
    input_params = data['input']
    #result =  predict.predict(input_params)
    #result =  predict(input_params)
    return jsonify({'prediction': 'test successful','input':input_params})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7778)