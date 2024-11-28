# backend/app.py

from flask import Flask, request, jsonify
from flask_cors import CORS
from simulation import run_simulation

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return "Monte Carlo Simulation Backend"

@app.route('/simulate', methods=['POST'])
def simulate():
    data = request.get_json()  # Get JSON data
    n_days = int(data['n_days'])
    result = run_simulation(n_days)
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
