import base64
import sys
import os
from flask import Flask, render_template,request
from keras.preprocessing.image import img_to_array, load_img
sys.path.append(os.path.abspath("./model"))

from load import init
app = Flask(__name__)
global model
model = init()

INPUT_HEIGHT = 64
INPUT_WIDTH = 64
INPUT_CHANNELS =3

#decoding an image from base64 into raw representation
def convertImage(imgData):
	imgData = imgData.decode('ascii').split(',')[1].encode('ascii')
	with open('output.png', 'wb') as output:
		output.write(base64.decodebytes(imgData))
	

@app.route('/')
def index():
	return render_template("index.html")

@app.route('/predict/',methods=['GET','POST'])
def predict():

	imgData = request.get_data()
	convertImage(imgData)

	x = load_img('output.png', target_size=(INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS))
	x = img_to_array(x)
	x = x / 255
	x = x.reshape(1, INPUT_WIDTH, INPUT_HEIGHT, INPUT_CHANNELS)

	proba = model.predict(x)

	dog_perc = int(proba[0][1] * 100)
	cat_perc = int(proba[0][0] * 100)

	if dog_perc > 40 and cat_perc > 40:
		response = "Both Dog ({}%) and Cat ({}%) present.".format(dog_perc, cat_perc)
	elif dog_perc > 40:
		response = "Only Dog ({}%) present.".format(dog_perc)
	elif cat_perc > 40:
		response = "Only Cat ({}%) present.".format(cat_perc)
	else:
		response = "Dog or Cat not found."

	return response
	

if __name__ == "__main__":
	port = int(os.environ.get('PORT', 5001))
	app.run(host='0.0.0.0', port=port)
