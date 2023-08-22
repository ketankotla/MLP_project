import os
from PIL import Image
import pandas as pd
import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import torch
import time
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

### Set Size for images
new_size= (500,400)

### Create dataframe with paths to all images
path_df = pd.DataFrame()
for kind in ["Data/benign", "Data/malignant", "Data/normal"]:
    df = pd.DataFrame({kind : os.listdir(kind)})
    path_df = pd.concat([path_df, df], axis = 1)


### Split dataset into train, test, and validation set, 60-20-20
train_df, test_df = train_test_split(path_df, test_size = 0.4)
valid_df, test_df = train_test_split(test_df, test_size = 0.5)


train_paths = []
train_y = []
for column in train_df.columns:
    for i in train_df[column]:
        if type(i) == type('a'):
            train_paths.append(os.path.join(column, i))
            if column == "Data/benign":
                train_y.append(0)
            elif column == "Data/malignant":
                train_y.append(2)
            elif column == "Data/normal":
                train_y.append(1)
            


test_paths = []
test_y = []
for column in test_df.columns:
    for i in test_df[column]:
        if type(i) == type('a'):
            test_paths.append(os.path.join(column, i))
            if column == "Data/benign":
                test_y.append(0)
            elif column == "Data/malignant":
                test_y.append(2)
            elif column == "Data/normal":
                test_y.append(1)


valid_paths = []
valid_y = []
for column in valid_df.columns:
    for i in valid_df[column]:
        if type(i) == type('a'):
            valid_paths.append(os.path.join(column, i))
            if column == "Data/benign":
                valid_y.append(0)
            elif column == "Data/malignant":
                valid_y.append(2)
            elif column == "Data/normal":
                valid_y.append(1)


BATCH_SIZE = 64


custom_train_transform = transforms.Compose([  
                                             transforms.ToTensor(),
                                             transforms.Normalize(mean=(0.5,), std=(0.5,))
])


class CancerDataset(Dataset):
    """Custom Dataset for loading cancer images"""

    def __init__(self, paths, y, transform=custom_train_transform):
        self.img_names = paths
        self.y = torch.Tensor(y).type(torch.LongTensor)
        self.transform = transform

    def __getitem__(self, index):
        img = Image.open(self.img_names[index])
        img = img.resize(new_size)
        
        if self.transform is not None:
            img = self.transform(img)
            
        
        label = self.y[index]
        return img, label

    def __len__(self):
        return len(self.y)


### Make Torch datasets and dataloaders for train, test, and validation set

train_dataset = CancerDataset(paths=train_paths, y = train_y,
                                    transform=custom_train_transform)


train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          drop_last=True)

test_dataset = CancerDataset(paths = test_paths, y = test_y,
                                   transform=custom_train_transform)

test_loader = DataLoader(dataset=test_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False)

valid_dataset = CancerDataset(paths = valid_paths, y = valid_y,
                                   transform=custom_train_transform)

valid_loader = DataLoader(dataset=valid_dataset,
                         batch_size=BATCH_SIZE,
                         shuffle=False)



### Create an MLP

class MLP(torch.nn.Module):

    def __init__(self, num_features, num_hidden_1, 
                 num_hidden_2, num_hidden_3, num_classes):
        super(MLP, self).__init__()
        
        self.num_classes = num_classes
        
        ### ADD ADDITIONAL LAYERS BELOW IF YOU LIKE
        self.linear_1 = torch.nn.Linear(num_features, num_hidden_1)
        self.bn1 = torch.nn.BatchNorm1d(self.linear_1.weight.size(0))
        self.linear_2 = torch.nn.Linear(num_hidden_1, num_hidden_2)
        self.bn2 = torch.nn.BatchNorm1d(self.linear_2.weight.size(0))
        self.linear_3 = torch.nn.Linear(num_hidden_2, num_hidden_3)
        self.bn3 = torch.nn.BatchNorm1d(self.linear_3.weight.size(0))
        self.linear_out = torch.nn.Linear(num_hidden_3, num_classes)
        
    def forward(self, x):
        
        ### MAKE SURE YOU CONNECT THE LAYERS PROPERLY IF YOU CHANGED
        ### ANYTHNG IN THE __init__ METHOD ABOVE       
        out = self.linear_1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.linear_2(out)
        out = self.bn2(out)
        out = F.relu(out)
        out = F.dropout(out, p=0.2, training=self.training)
        out = self.linear_3(out)
        out = self.bn3(out)
        out = F.relu(out)
        logits = self.linear_out(out)
        probas = F.softmax(logits, dim=1)
        return logits, probas

    

### Initialize the model
torch.manual_seed(0)
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


model = MLP(num_features=600000,
            num_hidden_1=75,
            num_hidden_2=50,
            num_hidden_3=25,
            num_classes=3)


model = model.to(DEVICE)




### Set optimizer to standard SGD
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)


NUM_EPOCHS = 20


def compute_accuracy_and_loss(model, data_loader, device):
    correct_pred, num_examples = 0, 0
    cross_entropy = 0.
    for i, (features, targets) in enumerate(data_loader):
            
        features = features.view(-1, 600000).to(device)
        targets = targets.to(device)

        logits, probas = model(features)
        cross_entropy += F.cross_entropy(logits, targets).item()
        _, predicted_labels = torch.max(probas, 1)
        num_examples += targets.size(0)
        correct_pred += (predicted_labels == targets).sum()
    return correct_pred.float()/num_examples * 100, cross_entropy/num_examples
    

start_time = time.time()
train_acc_lst, valid_acc_lst = [], []
train_loss_lst, valid_loss_lst = [], []


### Train the model

for epoch in range(NUM_EPOCHS):
    
    model.train()
    
    for batch_idx, (features, targets) in enumerate(train_loader):
    
        ### Prepare minibatch
        features = features.view(-1, 600000).to(DEVICE)
        targets = targets.to(DEVICE)
            
        ### Forward and back prop
        logits, probas = model(features)
        cost = F.cross_entropy(logits, targets)
        optimizer.zero_grad()
        
        cost.backward()
        
        ### Update parameters
        optimizer.step()
        
        ### Log
        if not batch_idx % 1:
            print (f'Epoch: {epoch+1:03d}/{NUM_EPOCHS:03d} | '
                   f'Batch {batch_idx:03d}/{len(train_loader):03d} |' 
                   f' Cost: {cost:.4f}')

    ### Evaluete model and log after each epoch
    model.eval()
    with torch.set_grad_enabled(False):
        train_acc, train_loss = compute_accuracy_and_loss(model, train_loader, device=DEVICE)
        valid_acc, valid_loss = compute_accuracy_and_loss(model, valid_loader, device=DEVICE)
        train_acc_lst.append(train_acc)
        valid_acc_lst.append(valid_acc)
        train_loss_lst.append(train_loss)
        valid_loss_lst.append(valid_loss)
        print(f'Epoch: {epoch+1:03d}/{NUM_EPOCHS:03d} Train Acc.: {train_acc:.2f}%'
              f' | Validation Acc.: {valid_acc:.2f}%')
        
    elapsed = (time.time() - start_time)/60
    print(f'Time elapsed: {elapsed:.2f} min')
  
elapsed = (time.time() - start_time)/60
print(f'Total Training Time: {elapsed:.2f} min')



### Evaluate model

model.eval()
with torch.set_grad_enabled(False): 
    test_acc, test_loss = compute_accuracy_and_loss(model, test_loader, DEVICE)
    print(f'Test accuracy: {test_acc:.2f}%')





