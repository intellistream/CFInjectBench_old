from models.T5_Model import T5 as T5_Model

def load_model(type: str):
    if type=='T5':
        return T5_Model
    else:
        raise Exception('Select the correct model type. Currently supporting only T5.')