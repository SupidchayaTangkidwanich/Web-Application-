from flask import Flask, render_template, request
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.efficientnet import preprocess_input

app = Flask(__name__)

Part = {p01: 'p01',p02: 'p02',p01: 'p01'}

import sys
sys.path.append('/supidchaya/Web-Application-/templates/Part.h5')

from efficientnet.layers import Swish, DropConnect
from efficientnet.model import ConvKernalInitializer
from tensorflow.keras.utils import get_custom_objects

get_custom_objects().update({
    'ConvKernalInitializer': ConvKernalInitializer,
    'Swish': Swish,
    'DropConnect':DropConnect
})

model1 = tf.keras.models.load_model('/supidchaya/Web-Application-/templates/Part.h5')


model1.make_predict_function()

# def predict_image1(img_path):
#     # Read the image and preprocess it
#     img = image.load_img(img_path, target_size=(150, 150))
#     x = image.img_to_array(img)
#     x = preprocess_input(x)
#     x = np.expand_dims(x, axis=0)
#     result = model1.predict(x)
#     return age[result.argmax()]

# def predict_image2(img_path):
#     # Read the image and preprocess it
#     img = image.load_img(img_path, target_size=(150, 150))
#     g = image.img_to_array(img)
#     g = preprocess_input(g)
#     g = np.expand_dims(g, axis=0)
#     result = model2.predict(g)
#     return gender[result.argmax()]
my_tuple = tuple(Part)

def predict_image1(img_path):
    # Read the image and preprocess it
    img = image.load_img(img_path, target_size=(150, 150))
    x = image.img_to_array(img)
    x = x.reshape((1,) + x.shape) 
    x /= 255.
    result = model1.predict(x)
    return my_tuple[int(result[0])]


# routes
@app.route('/')
def index():
    return render_template('upload.html')

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    if request.method == 'POST':
        # Read the uploaded image and save it to a temporary file
        file = request.files['image']
        img_path = 'static/p01.jpg'
        file.save(img_path)

        # Predict the age

        part_pred = predict_image1(img_path)

        # Render the prediction result
        return render_template('upload_completed.html', prediction1=part_pred)

if __name__ == '__main__':
    app.run(debug=True)