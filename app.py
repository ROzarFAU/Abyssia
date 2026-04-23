from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
import os
import tempfile
from backend import analyze_image

app = Flask(__name__, static_folder='static', template_folder='templates')
CORS(app)  

@app.route('/')
def index():
    """Serve the main frontend page"""
    return render_template('index.html')

@app.route('/analyze', methods=['POST'])
def analyze():
    """Endpoint that receives image and returns analysis results"""
    if 'image' not in request.files:
        return jsonify({'error': 'No image file provided'}), 400
    
    file = request.files['image']
    
    if file.filename == '':
        return jsonify({'error': 'Empty filename'}), 400
    
    with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp:
        file.save(tmp.name)
        temp_path = tmp.name
    
    try:
        result = analyze_image(temp_path)
        return jsonify(result)
    except Exception as e:
        return jsonify({'error': str(e)}), 500
    finally:
        if os.path.exists(temp_path):
            os.unlink(temp_path)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
