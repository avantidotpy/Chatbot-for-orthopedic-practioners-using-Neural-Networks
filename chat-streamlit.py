import streamlit as st

import json 
import numpy as np
from tensorflow import keras
from sklearn.preprocessing import LabelEncoder

import colorama 
colorama.init()
from colorama import Fore, Style, Back

import random
import pickle

with open("intents.json") as file:
    data = json.load(file)


def chat(inp):
    model = keras.models.load_model('chat_model')
    with open('tokenizer.pickle', 'rb') as handle:
        tokenizer = pickle.load(handle)
    with open('label_encoder.pickle', 'rb') as enc:
        lbl_encoder = pickle.load(enc)
    max_len = 20

    result = model.predict(keras.preprocessing.sequence.pad_sequences(tokenizer.texts_to_sequences([inp]),
                                             truncating='post', maxlen=max_len))
    tag = lbl_encoder.inverse_transform([np.argmax(result)])

    for i in data['intents']:
        if i['tag'] == tag:
            return np.random.choice(i['responses'])

st.title("MAXX ORTHO CHATBOT DEMO")
st.write("This is the prototype of the chatbot")
previous_messages =[]
user_input = st.text_input("You"," ")

if st.button("Send"):
    previous_messages.append(f"You: {user_input}")
    chatbot_reply = chat(user_input)
    previous_messages.append(f"Chatbot: {chatbot_reply}")
    
if previous_messages:
    st.write("Chat Log:")
    for message in previous_messages:
        if message.startswith("You:"):
            st.text_input(" ", value=message, key=message)
        else:
            num_lines = len(message.split("\n"))
            height = min(300, max(30, num_lines * 25))
            st.text_area(" ", value=message, height=height, key=message)
