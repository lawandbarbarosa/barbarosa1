import streamlit as st
import pickle
import numpy as np
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

# Load the trained model and the tokenizer
model = load_model('Text_emotion_classification_rnn.h5')  # Load the Keras model

with open('tokenizer.pkl', 'rb') as handle:
    tokenizer = pickle.load(handle)

# Load the max_length
with open('max_length.pkl', 'rb') as handle:
    max_length = pickle.load(handle)

# Define your emotion labels in the same order as your model's output
emotion_labels = ['sadness','anger','love','joy']

# Function to predict the emotion from the text
def prediction(Text):
    try:
        # Tokenize and pad the input text
        sequences = tokenizer.texts_to_sequences([Text])
        padded_sequences = pad_sequences(sequences, maxlen=max_length)
        
        # Predict the probabilities
        prediction_prob = model.predict(padded_sequences)
        
        # Get the predicted class index
        predicted_class = np.argmax(prediction_prob)
        
        # Check if predicted_class is within bounds
        if predicted_class < len(emotion_labels):
            return emotion_labels[predicted_class]
        else:
            return "Unknown"
    
    except Exception as e:
        st.error(f"Error during prediction: {e}")
        return "Error"

# Main function to define the Streamlit app
def main():
    st.title('Emotion Analysis Prediction')

    # Front end elements of the web page
    html_temp = """ 
    <div style ="background-color:yellow;padding:13px"> 
    <h1 style ="color:black;text-align:center;">Streamlit Emotion Analysis ML App </h1> 
    </div> 
    """
    st.markdown(html_temp, unsafe_allow_html=True)

    # Text input for the user's text
    Text = st.text_input('Your Text', 'Write Here')
    result = ''

    # When the 'Predict' button is clicked, make a prediction
    if st.button('Predict'):
        result = prediction(Text)
        st.success(f'The predicted emotion is: {result}')

if __name__ == '__main__':
    main()
