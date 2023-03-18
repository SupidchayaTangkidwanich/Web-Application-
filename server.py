from flask import Flask, render_template, request, send_file
from PIL import Image
import subprocess

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['POST'])
def upload():
    file = request.files['image']
    image = Image.open(file)
    image.save('static/p01.jpg')

    return render_template('upload_completed.html')

# @app.route('/playvid', methods=['POST'])
# def playvid():
#     text = request.form['text_input']
#     subprocess.run(["python", "/home/conceal/ConGUN-Code/ConGun-Visualization/python_script_person.py", "-uuid", text])
#     print(text)
#     return render_template('video_player.html')


if __name__ == '__main__':
    app.run(debug=True)