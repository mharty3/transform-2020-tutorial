from os import name
import requests
from io import BytesIO
import numpy as np
import joblib
from PIL import Image
from pprint import pprint

def img_to_arr(img):
    """
    Apply the same processing we used in training: greyscale and resize.
    """
    img = img.convert(mode='L').resize((32, 32))
    return np.asarray(img).ravel() / 255


def fetch_image(url):
    """
    Download an image from the web and pass to the image processing function.
    """
    r = requests.get(url)
    f = BytesIO(r.content)
    return Image.open(f) 


def predict_from_image(clf, img):
    """
    Classify an image.
    """
    arr = img_to_arr(img)
    X = np.atleast_2d(arr)
    probs = clf.predict_proba(X)
    result = {
        'class': clf.classes_[np.argmax(probs)],
        'prob': probs.max(),
        'classes': clf.classes_.tolist(),
        'probs': np.squeeze(probs).tolist(), # Must be serializable.
    }
    return result

if __name__ == '__main__':    
    import streamlit as st

    CLF = joblib.load('app-master/rf.gz')
    st.title("Fossil Net")
    url = st.text_input('Upload an Image URL')# 'https://images.fineartamerica.com/images-medium-large-5/23-trilobite-fossil-sinclair-stammersscience-photo-library.jpg'
    if url:
        img = fetch_image(url)
        result = predict_from_image(CLF, img)
        st.balloons()
        st.markdown(f'# {result["class"]}')
        st.markdown(f'p = {result["prob"].round(2)}')
        st.image(img)

        st.write(result)

    