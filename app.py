import os
from flask import Flask, render_template, request, jsonify
from keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

dic = {0 : 'Cat', 1 : 'Dog'}

model = load_model('model.h5')

def prepare_image(img):
  img = img.resize((224, 224))
  img = img.convert('RGB')
  img = image.img_to_array(img)
  img = np.expand_dims(img, axis=0)
  img = img/255
  return img

@app.route('/', methods=['GET', 'POST'])
def main():
  return render_template('index.html')

@app.route('/submit' , methods=['POST'])
def get_output():
  if 'my_image' not in request.files:
    return jsonify({'error' : 'No file part'})

  img = request.files['my_image']

  img_path = "static/" + img.filename
  img.save(img_path)

  prediction = predict_label(img_path)

  return render_template('index.html', prediction = prediction, img_path = img_path)

def predict_label(img_path):
  i = image.load_img(img_path, target_size=(100,100))
  i = prepare_image(i)
  prediction = model.predict(i)
  predicted_class = np.argmax(prediction, axis=1)[0]
  return dic[predicted_class]

if __name__ == "__main__":
    app.run(debug=True)