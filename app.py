from flask import Flask, request, render_template, jsonify

# from PIL import Image
from matplotlib import pyplot as plt
import numpy as np
#import face_recognition
import keras
from keras.models import load_model
import cv2
#from google.colab import files

from datetime import datetime
import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import MaxPooling2D

emotion_dict = {0: "Angry", 1: "Disgusted", 2: "Fearful", 3: "Happy", 4: "Neutral", 5: "Sad", 6: "Surprised"}

# Create the model
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(48, 48, 1)))
model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(128, kernel_size=(3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(1024, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(7, activation='softmax'))

# we have trained the model in Google Colab and all its weights have been saved
# Loading the save model's weights

model.load_weights('model.h5')

# Load the cascade for identifying face in the image
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  


def dectect_face(img):
    # Convert into grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 0, 255), 2)
        faces = img[y:y + h, x:x + w]
    
    return faces

def img_reshape(f):
    face_image = cv2.resize(f, (48,48))
    face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2GRAY)
    face_image = np.reshape(face_image, [1, face_image.shape[0], face_image.shape[1], 1])
    return face_image


# Read image
# img = cv2.imread('aziz2.jpg')
# plt.imshow(img)


# faces = dectect_face(img)
# predicted_class = np.argmax(model.predict(img_reshape(faces)))
# plt.imshow(faces)

# label_map = dict((k,v) for k,v in emotion_dict.items()) 
# predicted_label = label_map[predicted_class]
# predicted_label


def send_uploaded_file(filename=''):
    from flask import send_from_directory
    return send_from_directory(app.config["UPLOAD_FOLDER"], filename)


app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

UPLOAD_FOLDER = './uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    """Grabs the input values and uses them to make prediction"""
    # Read image

    if 'file1' not in request.files:
        return 'there is no file1 in form!'
    
    basedir = os.path.abspath(os.path.dirname(__file__))
    file1 = request.files['file1']

    file1.save(os.path.join(basedir, app.config['UPLOAD_FOLDER'], file1.filename))
    # path = os.path.join(app.config['UPLOAD_FOLDER'], file1.filename)
    # file1.save(path)
    # path = send_uploaded_file(file1.filename)
    img = cv2.imread('./uploads/' + file1.filename)

    faces = dectect_face(img)
    predicted_class = np.argmax(model.predict(img_reshape(faces)))
    # plt.imshow(faces)

    label_map = dict((k,v) for k,v in emotion_dict.items()) 
    predicted_label = label_map[predicted_class]
    # predicted_label = "aba"

    data = {
            "mood" : predicted_label,
            "id" : 1
        }
    return jsonify(data)
    # return render_template('index.html', prediction_text=f'Mode is = {predicted_label}')

if __name__ == "__main__":
    app.run()