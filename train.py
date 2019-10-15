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

# %% Difine input arguments from command line
parser = argparse.ArgumentParser(
    description='Train a neural network for flower catagorition detection',
    formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument('data_dir',
                    help='Image directory. Mandatory argument',
                    type=str)

parser.add_argument('--save_dir',
                    help='Checkpoints save directory.',
                    default='.',
                    type=str)

parser.add_argument('--arch',
                    help='Convolutional Network architeture',
                    default='vgg16',
                    type=str)

parser.add_argument('--learning_rate',
                    help='Learning rate',
                    default=0.001,
                    type=float)

parser.add_argument('--hiden_units',
                    help='hiden fully connected unit',
                    default=4096,
                    type=int)

parser.add_argument('--epochs', help='epochs', default=4, type=int)

parser.add_argument('--GPU', help='Train on GPU', action='store_true')

args = parser.parse_args()
print(args)

# %% Chose a device
GPU_available = torch.cuda.is_available()
if args.GPU == True and GPU_available:
    device = 'cuda'
else:
    device = 'cpu'

# %% Assign directory to variable
data_dir = args.data_dir
train_dir = data_dir + '/train'
valid_dir = data_dir + '/valid'
test_dir = data_dir + '/test'

# %% Loading data

data_transforms = {'train': None, 'valid': None, 'test': None}
image_datasets = {'train': None, 'valid': None, 'test': None}
dataloaders = {'train': None, 'valid': None, 'test': None}

data_transforms['train'] = transforms.Compose([transforms.RandomRotation(30),
                                               transforms.RandomResizedCrop(
                                                   224),
                                               transforms.RandomHorizontalFlip(),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])

data_transforms['valid'] = transforms.Compose([transforms.Resize(255),
                                               transforms.CenterCrop(224),
                                               transforms.ToTensor(),
                                               transforms.Normalize([0.485, 0.456, 0.406],
                                                                    [0.229, 0.224, 0.225])])

data_transforms['test'] = data_transforms['valid']

# Load the datasets with ImageFolder
image_datasets['train'] = datasets.ImageFolder(
    train_dir, transform=data_transforms['train'])
image_datasets['valid'] = datasets.ImageFolder(
    valid_dir, transform=data_transforms['valid'])
image_datasets['test'] = datasets.ImageFolder(
    test_dir, transform=data_transforms['test'])

# Using the image datasets and the trainforms, define the dataloaders
dataloaders['train'] = torch.utils.data.DataLoader(
    image_datasets['train'], batch_size=64, shuffle=True)
dataloaders['valid'] = torch.utils.data.DataLoader(
    image_datasets['valid'], batch_size=64)
dataloaders['test'] = torch.utils.data.DataLoader(
    image_datasets['test'], batch_size=64)

with open('cat_to_name.json', 'r') as f:
    cat_to_name = json.load(f)

# %% Define model
def define_model(args):
    network_str = args.arch
    # Check whether the model exist
    try:
        model = getattr(models, network_str)(pretrained=True)
    except AttributeError:
        print("No such network in the pytorch module! : " + network_str)
        exit()

    # Freeze parameters
    for param in model.parameters():
        param.requires_grad = False

    # Define a new classifier and assign to the model
    model.name = network_str
    in_features = model.classifier[0].in_features
    hidden_units = args.hiden_units
    classifier = nn.Sequential(OrderedDict([
        ('fc1', nn.Linear(in_features, hidden_units)),
        ('relu1', nn.ReLU()),
        ('dropout1', nn.Dropout(0.5)),
        ('fc2', nn.Linear(hidden_units, 102)),
        ('output', nn.LogSoftmax(dim=1))
    ]))
    model.classifier = classifier

    # Define loss function and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(),
                           lr=args.learning_rate)

    return model, criterion, optimizer
# %% Funtions 
# Define validation function
def validation_test(model, criterion, data_set: str, device):
    loss = 0
    accuracy = 0
    model.to(device)
    model.eval()
    with torch.no_grad():
        for inputs, labels in dataloaders[data_set]:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)

            batch_loss = criterion(logps, labels)
            loss += batch_loss.item()

            # Calculate accuracy
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()

    loss, accuracy = (
        loss/len(dataloaders[data_set]), accuracy/len(dataloaders[data_set]))

    return accuracy, loss

# Define train function
def train(model, criterion, optimizer, arg):
    epochs = 4
    steps = 0
    running_loss = 0
    print_every = int(len(dataloaders['train'])/5)
    model.to(device)
    print("Total number of minibatch: ", len(dataloaders['train']))
    print("Using device: {}".format(device))
    for epoch in range(epochs):
        for inputs, labels in dataloaders['train']:
            model.train()
            steps += 1
            # Move input and label tensors to the default device
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            logps = model.forward(inputs)
            loss = criterion(logps, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:

                accuracy, valid_loss = validation_test(
                    model, criterion, 'valid', device)

                print("Epoch: {}/{} |".format(epoch+1, epochs),
                      "Train loss {:3.3f} |".format(running_loss/print_every),
                      "Valid loss {:3.3f} |".format(valid_loss),
                      "Valid accuracy {:3.3f}".format(accuracy))

                running_loss = 0
                if (accuracy > 0.85):
                    print("Early stop at {}".format(accuracy))
                    break
        else:
            continue
        break
    return model

# %% Train
model, criterion, optimizer = define_model(args)
model = train(model, criterion, optimizer, args)
test_accuracy, test_loss = validation_test(model, criterion, 'test', device)
print("Test loss {:3.3f} |".format(test_loss),
        "Test accuracy {:3.3f}".format(test_accuracy))

# %% Saving the model
model.class_to_idx = image_datasets['train'].class_to_idx

checkpoint = {'architecture': model.name,
              'classifier': model.classifier,
              'class_to_idx': model.class_to_idx,
              'state_dict': model.state_dict()}

model.cpu()

if args.save_dir:
    torch.save(checkpoint, args.save_dir + '/checkpoint.pth')
else:
    torch.save(checkpoint, 'checkpoint.pth')
