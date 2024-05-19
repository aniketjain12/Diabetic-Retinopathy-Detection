import numpy as np
import torch
from train import DenseNet
import json
from PIL import Image
import streamlit as st
import requests
from streamlit_lottie import st_lottie


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # Loading and processing the image
    img = Image.open(image)

    # Resizing the image to 224x224 pixels
    img = img.resize((224, 224))

    # Converting to NumPy array and normalize
    np_image = np.array(img) / 255.0

    # Standardizing the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Transposing the image to the correct format (channels, height, width)
    np_image = np_image.transpose((2, 0, 1))

    # Convert to PyTorch tensor
    img_tensor = torch.from_numpy(np_image).float()

    return img_tensor

def predict(image, model_checkpoint='checkpoint.pth', category_names='output1.json', topk=2, device=None):
    # Loading the checkpoint and extract the model state dict
    checkpoint = torch.load(model_checkpoint)
    model_state_dict = checkpoint['state_dict']

    model = DenseNet()
    model.load_state_dict(model_state_dict)

    # Preprocessing the image
    img = process_image(image)

    img = torch.FloatTensor(img)

    img.unsqueeze_(0)

    # Setting the model to evaluation mode and move to the appropriate device
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    img = img.to(device)

    # Performing the forward pass
    with torch.no_grad():
        logps = model.forward(img)
 
    # Calculating probabilities and classes
    probabilities = torch.exp(logps)
    
    topk_prob, topk_class = probabilities.topk(topk, dim=1)

    
    topk_probs = topk_prob.tolist()[0]
    topk_classes = topk_class.tolist()[0]
    
    # Convert indices to class labels using class_to_idx
    idx_to_class = {val: key for key, val in checkpoint['class_to_idx'].items()}

    topk_classes = [idx_to_class[i] for i in topk_classes]

    if category_names:
       with open('output.json', 'r') as f:
        output = json.load(f)
    topk_names = [output[i] for i in topk_classes if i is not None]


    return topk_probs, topk_classes, topk_names

def main():
    
    st.set_page_config(page_title='Diabetic Retinopathy Detection', page_icon=':eye:', layout='wide', initial_sidebar_state='auto')
    
    def load_lottie_url(url: str):
        r = requests.get(url)
        if r.status_code != 200:
            return None
        return r.json()
    lottie_logo = load_lottie_url("https://lottie.host/d00d32fd-3013-4fe9-a094-19511ff212a4/YlngYyKyRv.json")
    lottie_ai = load_lottie_url("https://lottie.host/610e29f4-91d1-42f4-8608-cd510e78d623/21fFfswBAS.json")
    with st.container():

        left_column, right_column = st.columns([2, 1])
        with left_column:
            st_lottie(lottie_ai, height=300, key="ai")
        with right_column:    
            st.title('**_Diabetic Retinopathy Detection_**')
            st.write('Detecting diabetic retinopathy early with precision: Our model brings efficient screening to the forefront, enhancing patient care and vision preservation.')
            
    
    st.write("----------------------------------")
    with st.container():

        left_column, right_column = st.columns([2, 1])
        with left_column:
            
            st.header('Upload an image of eye for classification.')
            st.write("###")
        
            uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                image = Image.open(uploaded_file)
                st.image(image, caption='Uploaded Image', width=200, use_column_width="average")
            
            try:
                if st.button('Classify'):
                    topk_probs, topk_classes, topk_names = predict(uploaded_file)
                    st.write('Prediction:', topk_names)
            except:
                st.write('Please upload an image to classify.')
                
        with right_column:
            st_lottie(lottie_logo, height=300, key="logo")

    

           
if __name__ == '__main__':
    main()