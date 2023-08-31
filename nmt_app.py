
import streamlit as st
import re
import numpy as np
from tensorflow.keras.models import load_model
import string

# Replace this with the path to your saved model
model_path = 'nmt.h5'

# Load your model
seq2seq_model = load_model(model_path)

import pickle

with open("input_token_index.pkl", "rb") as f:
    input_token_index = pickle.load(f)

with open("target_token_index.pkl", "rb") as f:
    target_token_index = pickle.load(f)

with open("reverse_target_char_index.pkl", "rb") as f:
    reverse_target_char_index = pickle.load(f)

max_english_sen_len,max_french_sen_len  = 12,22
# Replace this with your actual translation function
def translate(input_sentence):
    punctuation = string.punctuation
    digits = string.digits
    input_sentence = input_sentence.lower()
    input_sentence = ''.join(char for char in input_sentence if char not in punctuation)
    input_sentence = ''.join(char for char in input_sentence if char not in digits)
    input_sentence = input_sentence.strip()
    input_sentence = re.sub(" +", " ", input_sentence)
    
    # Tokenize input sentence
    input_words = input_sentence.split()
    input_indices = [input_token_index.get(word, 0) for word in input_words]
    print(input_indices)
    # Create encoder input
    encoder_input_data = np.zeros((1, max_english_sen_len), dtype='float32')
    for t, index in enumerate(input_indices):
        encoder_input_data[0, t] = index
    
    # Initial decoder input
    decoder_input_data = np.zeros((1, 1), dtype='float32')
    decoder_input_data[0, 0] = target_token_index['<SOS>']
    
    # Translate using the model
    translation = ''
    for _ in range(max_french_sen_len):
        decoder_output = seq2seq_model.predict([encoder_input_data, decoder_input_data])
        predicted_token_index = np.argmax(decoder_output[0, -1, :])
        print(predicted_token_index)
        predicted_word = reverse_target_char_index.get(predicted_token_index, '<UNK>')
        
        if predicted_word == '<EOS>':
            break
        
        translation += predicted_word + ' '
        
        # Update decoder input for next iteration
        decoder_input_data = np.zeros((1, 1), dtype='float32')
        decoder_input_data[0, 0] = predicted_token_index
    
    return translation.strip()

def main():
    st.title("English to French Translator")

    # Input box for English sentence
    input_sentence = st.text_input("Enter an English sentence(max 12 words):", "")

    if input_sentence:
        # Translate the input text to French
        predicted_translation = translate(input_sentence)
        
        word_count = len(input_sentence.split())
        prediction = ""
        i = 0
        for word in predicted_translation.split():
          prediction+=word
          prediction+=" "
          i+=1
          if i==word_count:
            break
        
       # Display input sentence in a box
        st.text("Input English Sentence:")
        st.info(input_sentence)
        
        # Display translated sentence in a box
        st.text("Predicted French Translation:")
        st.success(prediction)
    # Adding an explanation section
    st.markdown("### How to Use")
    st.write("1. Enter an English sentence in the input box.")
    st.write("2. Just press Enter to see the predicted French translation.")
    
    # Adding a footer
    st.markdown("---")
    st.write("Powered by Streamlit • Model by Shray")    

if __name__ == "__main__":
    main()
