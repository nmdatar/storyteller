import streamlit as st
import openai
import os
import io
from keybert import KeyBERT
from dotenv import load_dotenv
from PIL import Image
from base64 import b64decode


load_dotenv()

def extract_keywords(text: str) -> list:
      kw_model = KeyBERT(model='all-MiniLM-L6-v2')
      model_outputs = kw_model.extract_keywords(text, keyphrase_ngram_range=(1, 3), use_maxsum=True, diversity=.9, top_n=10)
      keywords = [word[0] for word in model_outputs]
      return keywords

def generate_image(prompt: str):
      openai.api_key = os.environ['OPENAI_API_KEY']
      response = openai.Image.create(
            prompt=prompt, 
            n=1,
            size='256x256',
            response_format='b64_json',
      )
      
      data = b64decode(response['data'][0]['b64_json'])
      return data

def display_image(image_data):
      image = Image.open(io.BytesIO(image_data))
      st.image(image)

def download_image(image_data) -> None:
    # Provide the download link
    st.download_button(
        label='Download Generated Image to Storytell!',
        data=image_data,
        file_name='scene.png',
        mime='image/png'
    )

def run_pipeline(user_input: str) -> None:
      keywords = extract_keywords(text=user_input)

      # check if prompt is long enough to have extracted keywords
      if keywords:
            keywords = ", ".join(keywords)
      else: 
            keywords = user_input

      image_data = generate_image(prompt=keywords)
      display_image(image_data=image_data)
      download_image(image_data=image_data)



if __name__ == '__main__':
      st.write('Text-To-Image Storyteller Application')
      user_input = st.text_area('Enter text to storytell')
      if st.button("Tell the Story!"):
            if user_input:
                  run_pipeline(user_input=user_input)
                  st.success("Story has been told!")
            else:
                  st.warning("Please enter some text.")    