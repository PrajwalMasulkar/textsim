# 1. Library imports
import uvicorn
import numpy as np
import pickle
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from numpy import dot                                         
from numpy.linalg import norm
from fastapi import FastAPI
from pydantic import BaseModel
                         

# 2. Create the app object
app = FastAPI()


# 2. Class which describes text similarity measurements
class textsim(BaseModel):
    text1: str
    text2: str

module_url = "https://tfhub.dev/google/universal-sentence-encoder/4"
model = hub.load(module_url)

def find_similarity(sen1, sen2):
    messages = [sen1,sen2]
    message_embeddings = embed(messages)
    a = tf.make_ndarray(tf.make_tensor_proto(message_embeddings))
    return dot(a[0], a[1])/(norm(a[0])*norm(a[1])) 

def embed(input):
    return model(input)   



# 3. Expose the prediction functionality, make a prediction from the passed
#    JSON data and return the predicted similarity score
@app.post('/predict')
def predict_textsim(data: textsim):

    text1=data.text1
    text2=data.text2

    messages = [text1,text2]  
    message_embeddings = embed(messages)
    a = tf.make_ndarray(tf.make_tensor_proto(message_embeddings))
    b = str(dot(a[0], a[1])/(norm(a[0])*norm(a[1])))
    return {"similarity score ": b}


# 5. Run the API with uvicorn
#    Will run on http://127.0.0.1:8000
if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=8000)
    
#uvicorn app:app --reload
