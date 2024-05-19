import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
import os
import argparse
import json

def DenseNet(save_dir=os.path.dirname(os.path.abspath(__file__)), arch= 'resnet18', learning_rate= 0.002, hidden_units= 256,  device=None):
    if arch == "resnet18":
            model = models.resnet18(pretrained=True)
            torch.manual_seed(43)
            # Freezing parameters
            for param in model.parameters():
                param.requires_grad = False
                
            
    
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            classifier = nn.Sequential(nn.Linear(2048, hidden_units),
                               nn.ReLU(),
                               nn.Dropout(0.4),
                               #nn.Linear(4096, hidden_units),
                               nn.BatchNorm1d(hidden_units),
                               ##nn.ReLU(),
                               #nn.Dropout(0.3),
                               nn.Linear(hidden_units, 5),
                               nn.LogSoftmax(dim=1))

            model.classifier = classifier
            #criterion = nn.NLLLoss()
            criterion = nn.CrossEntropyLoss()
            #optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate, weight_decay = 1e-4)  
            optimizer = optim.SGD(model.classifier.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay = 1e-3)                     
            model.to(device)
            
    else:
        raise ValueError("Invalid Model Type")
    
    return model
    
def save(train_data, model, optimizer, save_file, epochs):
    checkpoint = {'input_size': 2048,
             'output_size': 5,
             'epochs': epochs,
             'state_dict': model.state_dict(),
             'class_to_idx': train_data.class_to_idx,
             'optimizer_state_dict':optimizer.state_dict}
    torch.save(checkpoint, save_file)

def load_checkpoint(filepath, model, optimizer, train_data):
    
        checkpoint = torch.load(filepath)
        input_size = checkpoint['input_size']
        output_size = checkpoint['output_size'] 
        epochs = checkpoint['epochs']
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict']())
        model.class_to_idx = train_data.class_to_idx
    
        return model

def test(testloader):
    
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        accuracy = 0
        test_loss = 0
        model.eval()
        
        for inputs, labels in testloader:
            inputs, labels = inputs.to(device), labels.to(device)
            logps = model.forward(inputs)
            batch_loss = criterion(logps, labels)
                    
            test_loss += batch_loss.item()
                    
            ps = torch.exp(logps)
            top_p, top_class = ps.topk(1, dim=1)
            equals = top_class == labels.view(*top_class.shape)
            accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
            print(f"Test loss: {test_loss/len(testloader):.3f}.."
                  f"Test accuracy: {accuracy/len(testloader):.3f}..")
            
        running_loss = 0
        model.train()
        return test_loss, accuracy

def train(data_dir, save_dir= os.path.dirname(os.path.abspath(__file__)), arch= 'resnet18', learning_rate= 0.002, hidden_units= 256, epochs= 2, device=None):
    torch.manual_seed(43)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    train_dir = data_dir + '/train'
    valid_dir = data_dir + '/valid'
    test_dir = data_dir + '/test'

    data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])

    test_transforms = transforms.Compose([transforms.Resize(225),
                                         transforms.CenterCrop(224),
                                         transforms.ToTensor(),
                                         transforms.Normalize([0.485, 0.456, 0.406],
                                                              [0.229, 0.224, 0.225])])


    validation_transforms = transforms.Compose([transforms.Resize(225),
                                                transforms.CenterCrop(224),
                                                transforms.ToTensor(),
                                                transforms.Normalize([0.485, 0.456, 0.406],
                                                                     [0.229, 0.224, 0.225])])


# Load the datasets with ImageFolder
#image_datasets = 
    train_data = datasets.ImageFolder(data_dir + '/train', transform=data_transforms)
    test_data = datasets.ImageFolder(data_dir + '/test', transform=test_transforms)
    validation_data = datasets.ImageFolder(data_dir + '/valid', transform=validation_transforms)

# Using the image datasets and the trainforms, define the dataloaders
#dataloaders = 
    trainloader = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)
    testloader = torch.utils.data.DataLoader(test_data, batch_size=32)
    validationloader = torch.utils.data.DataLoader(validation_data, batch_size=32)
    
    '''print("Trainloader:",trainloader)
    print("Validationloader:",validationloader)'''

    with open('output.json', 'r') as f:
        output = json.load(f)
    
     
    model = DenseNet(save_dir=save_dir, arch=arch, learning_rate=learning_rate, hidden_units=hidden_units)
    #criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.classifier.parameters(), lr=learning_rate, weight_decay = 1e-4)
    optimizer = optim.SGD(model.classifier.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay = 1e-3)
    model.to(device)

    steps = 0 
    running_loss = 0
    print_every = 5
    best_validation_loss = float('inf')
    patience = 3
    no_improvement_count = 0
    
    for epoch in range(epochs):
        
        for inputs, labels in trainloader:
          
            steps += 1
            inputs, labels = inputs.to(device), labels.to(device)
        
            optimizer.zero_grad()
        
            logps = model.forward(inputs)
            loss = criterion(logps, labels)

            l2_reg = torch.tensor(1e-5, requires_grad=False).to(device)
            for param in model.parameters():
                l2_reg += torch.sum(param ** 2)

            loss += 1e-4 * l2_reg  # You can adjust the regularization strength
            
            
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
        
            if steps % print_every == 0:
                test_loss = 0
                accuracy = 0
                model.eval()
            
                with torch.no_grad():
                    for inputs, labels in validationloader:

                        inputs, labels = inputs.to(device), labels.to(device)
                        logps = model.forward(inputs)
                        batch_loss = criterion(logps, labels)
                    
                        test_loss += batch_loss.item()
                    
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()
                    
                print(f"Epoch {epoch+1}/{epoch}.."
                     f"Train loss: {running_loss/print_every:.3f}.."
                    f"Validation loss: {test_loss/len(validationloader):.3f}.."
                    f"Validation accuracy: {accuracy/len(validationloader):.3f}..")
            
                running_loss = 0
                train_loss = running_loss / print_every
                validation_loss /= len(validationloader)
                accuracy /= len(validationloader)
                # Check for early stopping
                if validation_loss < best_validation_loss:
                    best_validation_loss = validation_loss
                    no_improvement_count = 0
                else:
                    no_improvement_count += 1
                    if no_improvement_count >= patience:
                        print("Validation loss hasn't improved in the last", patience, "epochs. Early stopping.")
                        break
                model.train()

    return model 

if __name__ =='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_dir")
    parser.add_argument("--save_dir", help= "Save Directory" ,action= 'store', default= os.path.dirname(os.path.abspath(__file__)))
    parser.add_argument("--arch", help= "Model Name",action= 'store', default= 'resnet18')
    parser.add_argument("--learning_rate", help="Learning Rate", action= 'store', default=0.002)
    parser.add_argument("--hidden_units", help="Hidden Layers", action= 'store', default=256)
    parser.add_argument("--epochs", help="Number of Ephocs", action = 'store', default=2)
    parser.add_argument("--gpu", help="GPU(cuda)", action = 'store_true', default=None)
    
    args = parser.parse_args()
    model = train(args.data_dir, save_dir=args.save_dir, arch=args.arch, learning_rate=args.learning_rate, device=args.gpu, hidden_units=args.hidden_units,epochs=int(args.epochs))
    save_file = os.path.join(args.save_dir, 'checkpoint.pth')
    train_dir = args.data_dir + '/train'
    data_transforms = transforms.Compose([transforms.RandomRotation(30),
                                          transforms.RandomResizedCrop(224),
                                          transforms.RandomHorizontalFlip(),
                                          transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
                                          transforms.ToTensor(),
                                          transforms.Normalize([0.485, 0.456, 0.406],
                                                               [0.229, 0.224, 0.225])])
   
    train_data = datasets.ImageFolder(args.data_dir + '/train', transform=data_transforms)
    #criterion = nn.NLLLoss()
    criterion = nn.CrossEntropyLoss()
    #optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate, weight_decay = 1e-4)
    optimizer = optim.SGD(model.classifier.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay = 1e-3)
    save(train_data, model, optimizer, save_file, int(args.epochs))
    print("Training Complete. Checkpoint saved at {}".format(save_file))
        
        