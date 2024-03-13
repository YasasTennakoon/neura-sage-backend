from flask import Flask
app = Flask(__name__)

@app.route('/')
def index():
    return 'Welcome to my Flask API'

@app.route('/api/data')
def get_data():
    # Your logic to retrieve and return data goes here
    return jsonify({'key': 'value'})


if __name__ == '__main__':
    app.run(debug=True)

