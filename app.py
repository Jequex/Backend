import numpy as np
import pickle

from tensorflow.keras.models import load_model
from tensorflow.keras.utils import load_img, img_to_array
from flask import Flask, request
from flask_cors import CORS


app = Flask(__name__)
CORS(app)

MODEL_PATH = './classifier.h5'
model = load_model(MODEL_PATH)
model.make_predict_function()

with open('ResultsMap.pkl', 'rb') as f:
  ResultsMap = pickle.load(f)

@app.route('/upload', methods = ['POST'])
def upload_file():
  if request.method == 'POST':
    print('Uploading', request.files)
    testImage = request.files['file']

    imgPath = 'static/images/' + testImage.filename

    testImage.save(imgPath)

    imgPathNew = R"./"+imgPath

    test_image=load_img(imgPathNew,target_size=(128, 128))
    test_image=img_to_array(test_image)
    test_image=np.expand_dims(test_image,axis=0)
    result=model.predict(test_image,verbose=0)

    print(ResultsMap[np.argmax(result)])
    # f.save(secure_filename(f.filename))
    return {'img': 'http://localhost:5000/'+imgPath, 'prediction': ResultsMap[np.argmax(result)], 'status': 'success'}
  
if __name__ == '__main__':
  app.run(debug = True)