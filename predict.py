#! python

# %% import liberary
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import json
import argparse
from collections import OrderedDict
import PIL

# %% Difine input arguments from command line
parser = argparse.ArgumentParser(
    description = 'Train a neural network for flower catagorition detection',
    formatter_class = argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('image_path',
                    help = 'Image directory. Mandatory argument',
                    type = str)

parser.add_argument('load_model_dir',
                    help = 'Checkpoints load directory.',
                    type = str)

parser.add_argument('--topk',
                    help = 'Number of top most interest.',
                    default = 5,
                    type = int)

parser.add_argument('--GPU', help = 'Train on GPU', action = 'store_true')

args = parser.parse_args()

if args.GPU == True:
    device = 'cuda'
else:
    device = 'cpu'

# %% Mapping directory name to label
with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)


def load_checkpoint( filepath):
    
    # Here, we do not train any more. It is OK to use CPU.
    checkpoint = torch.load(filepath)
    network_str = checkpoint['architecture']
    model = getattr(models, network_str)(pretrained=True)
    model.cpu()
    
    for params in model.parameters(): 
        params.requires_grad = False
        
    model.class_to_idx = checkpoint['class_to_idx']
    model.classifier = checkpoint['classifier']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model

model = load_checkpoint(args.load_model_dir+"/checkpoint.pth")

# %% Define some functions
def process_image(image_path):
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    orig_img  = PIL.Image.open(image_path)
    
    transfrom_img = transforms.Compose([transforms.Resize(256),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],[0.229, 0.224, 0.225])])
    
    pytorch_tensor = transfrom_img(orig_img)
    return pytorch_tensor

def predict(image_path, model, topk = 5):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    device = 'cpu'
    
    # TODO: Implement the code to predict the class from an image file
    model.to(device)
    model.eval()
    img = process_image(image_path)
    img = img[None, :, :, :]
    img = img.float()
    img = img.to(device)
    
    with torch.no_grad():
        output = model(img)
        
    proba = torch.exp(output)
    
    probs, label = proba.topk(topk)
    
    index_to_cat = dict([(value, key) for key, value in model.class_to_idx.items()])
    index_to_name = dict([(key, cat_to_name[value]) for key, value in index_to_cat.items()])

    probs = probs.squeeze_().tolist()
    topk  = label.squeeze_().tolist()
    topk_cat = [ index_to_cat[i] for i in topk ]
    labs_name = [ index_to_name[i] for i in topk ]
    
    return probs, topk_cat, labs_name

# %% Calculate the model
probs, topk, labs_name = predict(args.image_path, model, args.topk)

print("Probability:      ", probs)
print("Category number:  ", topk)
print("Category to name: ", labs_name)