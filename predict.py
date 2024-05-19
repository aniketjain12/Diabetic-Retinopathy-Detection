import numpy as np
import torch
from train import DenseNet
import argparse
import json
from PIL import Image


def process_image(image):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''

    # Loading and processing the image
    img = Image.open(image)

    # Resizing the image to 224x224 pixels
    img = img.resize((224, 224))

    # Converting to NumPy array and normalizing
    np_image = np.array(img) / 255.0

    # Standardizing the image
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    np_image = (np_image - mean) / std

    # Transposing the image to the correct format (channels, height, width)
    np_image = np_image.transpose((2, 0, 1))

    # Converting to PyTorch tensor
    img_tensor = torch.from_numpy(np_image).float()

    return img_tensor

def predict(image_path, model_checkpoint, category_names='output.json', topk=5, device=None):
    # Loading the checkpoint and extract the model state dict
    checkpoint = torch.load(model_checkpoint)
    model_state_dict = checkpoint['state_dict']

    model = DenseNet()
    model.load_state_dict(model_state_dict)

    # Preprocessing the image
    img = process_image(image_path)
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

    # Converting indices to class labels using class_to_idx
    idx_to_class = {val: key for key, val in checkpoint['class_to_idx'].items()}
    topk_classes = [idx_to_class[i] for i in topk_classes]
    if category_names:
       with open('output1.json', 'r') as f:
        output1 = json.load(f)
    topk_names = [output1[i] for i in topk_classes if i is not None]


    return topk_probs, topk_classes, topk_names
if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("image_path", help='Enter Image', action='store')
    parser.add_argument("--checkpoint", help='Checkpoint file', action='store', default='checkpoint.pth')
    parser.add_argument("--topk", help='Top values', action='store', default=2, type=int)
    parser.add_argument("--category_name", help='Name of category with json', action='store', default='output1.json', type=str)
    parser.add_argument("--gpu", help='GPU(cuda)', action='store')
    
    args = parser.parse_args()
    topk_probs, topk_classes, topk_names = predict(args.image_path, model_checkpoint=args.checkpoint, category_names = args.category_name, topk=args.topk, device=args.gpu)
    
    zipped_result = zip(topk_classes, topk_names, topk_probs)
    result_list = list(zipped_result)
    for classes, names, probs in result_list:
        print(f"Your Eye is: {names}... Probability of Diabetic Retinopathy in your eye is: {probs}")


