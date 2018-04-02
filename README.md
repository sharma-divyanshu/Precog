# PrecogTask

A Python based face detection/classification web application developed as a task for Precog, IIIT Delhi.

<b>Overview:</b>

The web application inputs an image from the user, draws a boundary box around any faces in the image, and evaluates whether they belong to Arvind Kejriwal and Narendra Modi.

<b>Working/Libraries Used:</b>

<ol><li>The system has been deployed as a web application using Flask. The application runs @ 127.0.0.1:4555
<li>Face detection is performed using Haar cascades, part of the OpenCV library.
<li>Once faces are detected, prediction whether the faces belong to Modi or Kejriwal is made based on two CNNs trained on a dataset of images belonging to the two individuals.
