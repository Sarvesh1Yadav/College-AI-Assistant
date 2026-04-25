from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)

@app.route('/')
def index():
    return send_file('index.html')

@app.route('/ask', methods=['POST'])
def ask():
    try:
        data = request.json
        question = data.get('question', '')
        if not question.strip():
            return jsonify({'error': 'Please enter a question'}), 400

        from app.query_analyzer import handle_query
        result = handle_query(question)
        return jsonify({
            'answer': result.get('answer', 'Sorry, I could not find an answer to your question.'),
            'sources': result.get('sources', [])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 7860))
    app.run(host='0.0.0.0', port=port, debug=True)
