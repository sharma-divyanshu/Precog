# PrecogTask

A Python based face detection/classification web application.

<b>Overview:</b>

The web application inputs an image from the user, draws a boundary box around any faces in the image, and evaluates whether they belong to Arvind Kejriwal or Narendra Modi.

<b>Libraries Used:</b>

<ol><li>The system has been deployed as a web application using Flask. The application runs @ 127.0.0.1:5000
<li>Face detection is performed using Haar cascades, part of the OpenCV library.
<li>Once faces are detected, prediction whether the faces belong to Modi or Kejriwal is made based on two CNNs trained on a dataset of images belonging to the two individuals.</ol>

<b>Process/Working</b>

The initial task was divided into 4 parts:

1. <b>Creation of a web application:</b> This was achieved by creating a Flask application, with basic HTML pages `index.html` that provides a form for uploading the image, and `upload.html` that displays the results.
2. <b>Face Detection:</b> Performed using OpenCV (Haar cascades) using a pre-trained face detection model. The function returns the number of faces as well as prints the rectangle boundary boxes around the detected faces on the images.
3. <b>Dataset collection:</b> Used Fatkun batch downloader to download 500-700 images each of Arvind Kejriwal and Narendra Modi, that were manually separated into testing and training sets for <i>positive samples</i>. For <i>negative samples</i>, downloaded 500 images of random faces, along with the use of images of the other class to improve accuracy. The dataset was imported to a MongoDB collection.
4. <b>Image Classification:</b> Used two Convolutional Neural Networks to detect the presence of Kejriwal/Modi in the images. The models were trained locally, and are available inside the `classifiers` directory.

<b>Directory Structure:</b>

```
+-classifiers  //saved CNN models
  --kejriwal.h5
  --modi.h5
+-CNN  //CNN train/test algorithm
  --cnn.py
+-dataset 
  +-precog_dataset  //MongoDB dump
    --fs.chunks.bson
    --fs.chunks.metadata.bson
    --fs.files.bson
    --fs.files.metadata.json
  --dataset.rar  //Dataset compressed archive
+-haarcascades  //haarcascade .xml model
+-static
  +-etc
    +-images
      +-uploaded  //contains user uploaded images
  +-images  //contains images with faces detected and marked
  +-styles  //contains css
+-templates
  --index.html
  --upload.html
--database.py  //used for uploading dataset images to MongoDB collection
--repo.py  //contains face detection/recognition functions
--Run.py  //contains Flask code
--__init__.py  //Runs the WSGI application
--requirements.txt  //specifies dependencies
```
<b>Deployment:</b>

The application was deployed on a Google Cloud Platform virtual machine running Ubuntu 14.04 using mod_wsgi, as prevalent app hosting websites do not provide complete support for OpenCV/Tensorflow and related dependencies.

The web app is live [here](http://35.187.242.13/)

The app can be run locally at `localhost:5000` by executing the `__init__.py` file.
 
<b>Area of Improvement:</b>

The face detection process can be optimized by implementing a condition which runs the CNN prediction *only if* faces are detected by the haar cascade. However, by experimentation, it has been observed that the CNNs have a higher probability to recognize faces and predict the correct person, whereas the haar classifier often misses faces due to obstacles in the image. Therefore, the condition has not been implemented. For now, the prediction models run even if the classifier does not detect any faces.

<b>Screenshots:</b>

<p><img src="https://github.com/sharma-divyanshu/Precog/blob/master/screenshots/2.PNG?raw=true"></p>
<p><img src="https://github.com/sharma-divyanshu/Precog/blob/master/screenshots/1.PNG?raw=true"></p>
<p><img src="https://github.com/sharma-divyanshu/Precog/blob/master/screenshots/3.PNG?raw=true"></p>
<p><img src="https://github.com/sharma-divyanshu/Precog/blob/master/screenshots/4.PNG?raw=true"></p>
