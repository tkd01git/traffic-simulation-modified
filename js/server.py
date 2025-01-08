from flask import Flask, request, jsonify
from flask_cors import CORS  # 追加
import subprocess

app = Flask(__name__)
CORS(app)  # すべてのオリジンからのリクエストを許可

@app.route('/run-script', methods=['POST'])
def run_script():
    print("Request received: /run-script")
    data = request.get_json()
    input_filename = data.get('inputFilename')
    output_filename = data.get('outputFilename')
    
    try:
        print(f"Running script with input: {input_filename}, output: {output_filename}")
        result = subprocess.run(
            ['python', 'excelcompile.py', input_filename, output_filename],
            capture_output=True,
            text=True,
            encoding='utf-8'
        )
        if result.returncode == 0:
            print("Script executed successfully")
            return jsonify({"message": "Python script executed successfully", "output": result.stdout})
        else:
            print(f"Script failed with error: {result.stderr}")
            return jsonify({"message": "Python script failed", "error": result.stderr}), 500
    except Exception as e:
        print(f"Error executing Python script: {str(e)}")
        return jsonify({"message": "Error executing Python script", "error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
