import os,subprocess
import glob
import time

# External modules
import numpy as np
import pandas as pd
#from tqdm import tqdm

# Librosa for audio


# Plotting modules


from keras.preprocessing import image
from keras.applications import VGG16
from keras.models import load_model


#import werkzeug
from flask import Flask,render_template, url_for, request, redirect, send_from_directory
from flask_restful import reqparse, abort, Api, Resource
from flask import jsonify 
from werkzeug.utils import secure_filename



UPLOAD_FOLDER = '../upload/'
ALLOWED_EXTENSIONS = set(['txt', 'pdf', 'png', 'jpg', 'jpeg', 'gif','wav','mp4','mp3'])

app=Flask(__name__)
#api=Api(app)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route("/")

def hello():
    return render_template('index.html')


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route("/fun",methods=['GET','POST'])

def fun():

    if request.method == 'POST':
        # check if the post request has the file part
        if 'file' not in request.files:
            #flash('No file part')
            #return request.url/shivam1
            return "shivam1"

        file = request.files['file']

        # if user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            #flash('No selected file')
            #return request.url/shivam
            return "shivam2"

        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))

            return redirect(url_for('uploaded_file', filename=filename))
            #return "shivam3"


@app.route('/predict/<filename>',methods=['GET', 'POST'])

def uploaded_file(filename):

    file = filename
    name = file[:file.rfind(".")]
    #file = send_from_directory(app.config['UPLOAD_FOLDER'], filename)
    #os.chdir("/home/slytherin/Downloads/app+apis/")
    conv_base = VGG16(weights='imagenet',
    include_top=False,
    input_shape=(128, 128,3))
    dir = "/home/slytherin/Downloads/app+api/upload"

    model = load_model('pikakpi.h5')

    img_path = dir+"/"+name+".jpg"
   
    #print(img_path)

    img = image.load_img(img_path, target_size=(128, 128))
    x = image.img_to_array(img)


    x = x/255.

    f=conv_base.predict(x.reshape(1,128, 128,3))
    clas = model.predict_classes(f.reshape(1,4*4*512))
    clas = str(clas)
    print(clas)
    return clas


if __name__=='__main__':
    app.run(host="192.168.43.43", port=5010, debug=False)
    
