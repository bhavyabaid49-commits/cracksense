import os
from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
from model_handler import predict_image_severity

app = Flask(__name__)
CORS(app)   # <-- Enable CORS for all routes (keep this only once)

# Create uploads folder if missing
UPLOAD_FOLDER = 'uploads'
if not os.path.exists(UPLOAD_FOLDER):
    os.makedirs(UPLOAD_FOLDER)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/', methods=['GET'])
def home():
    html_form = '''
    <!doctype html>
    <title>CrackSense Local Test</title>
    <h2>CrackSense AI Backend - Test Upload</h2>
    <form method=post action="/predict" enctype=multipart/form-data>
      <input type=file name=file>
      <input type=submit value=Upload>
    </form>
    '''
    return render_template_string(html_form)

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No file selected for uploading"}), 400

    if file:
        # Save uploaded image
        file_path = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(file_path)

        # AI prediction
        ai_result = predict_image_severity(file_path)

        if "error" in ai_result:
            return jsonify(ai_result), 500

        detected_severity = ai_result["severity"]
        confidence = ai_result["confidence"]

        # Engineering rules
        if detected_severity == "hairline":
            material = "Polyurethane Injection Grout"
            quantity = "0.5 kg per meter"
            cost_per_meter = 400
            risk_level = "Low"
        elif detected_severity == "moderate":
            material = "Epoxy Resin Injection"
            quantity = "1.2 kg per meter"
            cost_per_meter = 1200
            risk_level = "Medium"
        else:  # critical
            material = "Micro-Concrete & Carbon Fiber Wrap"
            quantity = "5.0 kg per meter"
            cost_per_meter = 4500
            risk_level = "High"

        # Cleanup temp file
        if os.path.exists(file_path):
            os.remove(file_path)

        result = {
            "project_name": "CrackSense",
            "analysis": {
                "crack_type": "Structural Surface Anomaly",
                "severity": detected_severity,
                "confidence_score": confidence,
                "risk_level": risk_level
            },
            "repair_estimate": {
                "recommended_material": material,
                "estimated_quantity": quantity,
                "approximate_cost_inr": cost_per_meter
            }
        }
        return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, port=5000)