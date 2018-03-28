import os
from flask import Flask, render_template, request
from repo import detect_faces, isModi, isKejriwal

# creating the webpage

app = Flask(__name__)

app_dir = os.path.dirname(os.path.abspath("__file__"))

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=['POST'])
def upload():
    target = os.path.join(app_dir, 'images/')
    if not os.path.isdir(target):
        os.mkdir(target)
    file = request.files["file"]
    filename = file.filename
    location = "".join([target, filename])
    file.save(location)
    nb_faces, source = detect_faces(location, filename)
    if nb_faces == 0:
        nb_faces = "None"
    else:
        modi = isModi(location, filename)
        kejriwal = isKejriwal(location, filename)
    source = str(source)
    return render_template("upload.html", nb_faces=nb_faces, source=source, modi=modi, kejriwal=kejriwal)

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=4555)