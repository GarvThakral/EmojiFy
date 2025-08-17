import streamlit as st
import pandas as pd
import tensorflow as tf
import numpy as np
import json

@st.cache_resource
def load_model():

    with open("test.json" , 'r') as f:
        lines = f.readlines()
        print(type(lines[0]))
        print("Starting ...")
        print(lines[0][:10])
        print(lines[0][-10:])
        print("...ending")
        word_to_index_map = json.loads(lines[0])
    model = tf.keras.models.load_model('my_model.keras')
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    return model , word_to_index_map

model , word_to_index_map = load_model()

def sentence_to_indices(sentence,maxLength = 10):
    m = sentence.shape[0]
    finalArr = np.zeros((m,maxLength),dtype = 'int')
    for i in range(m):
        ind = 0
        for j in sentence[i].lower().split():
            if j in word_to_index_map.keys():
                finalArr[i][ind] = word_to_index_map[j]
                ind+=1
    return finalArr





st.text_input("Your sentence ", key="sentence")

if st.button("Generate emoji" , type = "primary"):
    sentence = st.session_state.sentence
    sentenceArr = np.array([sentence],dtype='str')
    emojiIndx = np.argmax(model.predict(sentence_to_indices(sentenceArr)))
    print(emojiIndx)
    label_to_emoji = {0:"❤️",1:"⚾",2:"😀",3:"😔",4:"🍴"}
    st.write(f"{sentence} {label_to_emoji[emojiIndx]}")



