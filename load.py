from keras.models import model_from_json

def init(): 
	json_file = open('model/model.json', 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)

	loaded_model.load_weights("model/model.h5")
	print("Loaded Model from disk")

	return loaded_model