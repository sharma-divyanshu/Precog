import os
from flask import Flask, render_template, request
from werkzeug.utils import secure_filename
from repo import detect_faces, isModi, isKejriwal

# creating the webpage

app = Flask(__name__)

app_dir = os.path.dirname(os.path.abspath("__file__"))

UPLOAD_FOLDER = '/static/etc/uploaded'

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=['GET', 'POST'])
def upload():
    target = os.path.join(app_dir, 'static/etc/uploaded/')
    if not os.path.isdir(target):
        os.mkdir(target)
    file = request.files['file']
    filename = secure_filename(file.filename)
    location = "".join([target, file.filename])
    file.save(location)
    nb_faces = 'None'
    nb_faces, source = detect_faces(location, filename)
    modi = 'No'
    kejriwal = 'No'
#    if nb_faces == 0:
#        nb_faces = "None"
#    elif nb_faces==1:
#        modi = isModi(location, filename)
#        if modi=='No':
#            kejriwal = isKejriwal(location, filename)
#    else:
    modi = isModi(location, filename)
    kejriwal = isKejriwal(location, filename)
    if (modi == 'Yes') or (kejriwal == 'Yes') and (nb_faces == 0):
        nb_faces = 'Yes'
    
    source = str(source)
    return render_template("upload.html", nb_faces=nb_faces, source=source, modi=modi, kejriwal=kejriwal)

if __name__ == "__main__":
    app.run()