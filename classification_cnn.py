import pandas as pd 
# import cv2
import os
import matplotlib.pyplot as plt
import numpy as np
np.set_printoptions(precision=2)
import seaborn as sns

from sklearn import metrics
from sklearn.metrics import confusion_matrix, precision_score
from sklearn.metrics import recall_score, f1_score, accuracy_score
from sklearn.model_selection import cross_val_score

from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import GridSearchCV
from sklearn import preprocessing

from sklearn.model_selection import train_test_split

from sklearn.feature_selection import SelectFromModel
from sklearn.svm import LinearSVC
from sklearn.linear_model import SGDClassifier
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.preprocessing import LabelEncoder

import time
import copy

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import PIL

from PIL import Image
from collections import OrderedDict


import torch
from torch import nn, optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data.dataset import Dataset

class DatasetFromImages(Dataset):
    def __init__(self, data_frame, img_dir, transforms=None):
        """
        Args:
            csv_path (string): path to csv file
            img_path (string): path to the folder where images are
            transform: pytorch transforms for transforms and tensor conversion
        """
        # Transforms
        self.transforms = transforms
        # Read the csv file
        self.data_info = data_frame
        self.image_arr = np.asarray(img_dir+'/'+self.data_info['image_id']+'.jpg')
        
        self.encoder = preprocessing.LabelEncoder()
        self.label_arr = torch.from_numpy(self.encoder.fit_transform(self.data_info['dx'])).long()
        self.classes = self.encoder.classes_
        self.data_len = len(self.data_info)

    def __getitem__(self, index):
        # Get image name from the pandas df
        single_image_name = self.image_arr[index]
        # Open image
        img_as_img = Image.open(single_image_name)
        # Transform image to tensor
        if self.transforms is not None:
            img_as_tensor = self.transforms(img_as_img)
        # Get label
        single_image_label = self.label_arr[index]
        return (img_as_tensor, single_image_label)

    def __len__(self):
        return self.data_len
    
def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    return device 


def prepare_datasets(part=False):
    img_dir = 'images'
    csv_path = 'data/HAM10000_metadata.csv'
    
    df = pd.read_csv(csv_path)
    if part:
        gb = df.groupby('dx')    
        df_gb = [gb.get_group(x)[:100] for x in gb.groups]
        df = pd.concat(df_gb)
        
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df['dx'])
    train_df, valid_df = train_test_split(train_df, test_size=0.2)
    df_dict = {'train': train_df, 'valid': valid_df, 'test': test_df}
    
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomRotation(30),
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ]),
        'valid': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], 
                                 [0.229, 0.224, 0.225])
        ])
    }
    data_transforms['test'] = data_transforms['valid']

    image_datasets = {x: DatasetFromImages(df_dict[x], img_dir, data_transforms[x])  \
                      for x in ['train', 'valid', 'test']}
    
    # Using the image datasets and the trainforms, define the dataloaders
    batch_size = 64
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                 shuffle=True, num_workers=0)
                  for x in ['train', 'valid', 'test']}
    
    class_names = image_datasets['train'].classes
    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'valid', 'test']}
    
    return image_datasets, dataloaders, class_names, dataset_sizes

    
def create_model(model_name):
    if model_name == 'densenet':
        model = models.densenet161(pretrained=True)
        num_in_features = 2208
        # print(model)
    elif model_name == 'vgg':
        model = models.vgg19(pretrained=True)
        num_in_features = 25088
        # print(model.classifier)
    else:
        print('Unknown model')
        return None
    for param in model.parameters():
        param.requires_grad = False
    return model, num_in_features

def bind_model_classifier(model_name, classifier):
    if model_name == 'densenet':
        model.classifier = classifier
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adadelta(model.parameters()) # Adadelta #weight optim.Adam(model.parameters(), lr=0.001, momentum=0.9)
        #optimizer_conv = optim.SGD(model.parameters(), lr=0.0001, weight_decay=0.001, momentum=0.9)
        sched = optim.lr_scheduler.StepLR(optimizer, step_size=4)
    elif model_name == 'vgg':
        model.classifier = classifier
        criterion = nn.NLLLoss()
        optimizer = optim.Adam(model.classifier.parameters(), lr=0.0001)
        sched = lr_scheduler.StepLR(optimizer, step_size=4, gamma=0.1)
    else:
        pass
    return criterion, optimizer, sched
        
        
def create_classifier(num_in_features, hidden_layers, num_out_features):       
    classifier = nn.Sequential()
    if hidden_layers == None:
        classifier.add_module('fc0', nn.Linear(num_in_features, 102))
    else:
        layer_sizes = zip(hidden_layers[:-1], hidden_layers[1:])
        classifier.add_module('fc0', nn.Linear(num_in_features, hidden_layers[0]))
        classifier.add_module('relu0', nn.ReLU())
        classifier.add_module('drop0', nn.Dropout(.6))
        classifier.add_module('relu1', nn.ReLU())
        classifier.add_module('drop1', nn.Dropout(.5))
        for i, (h1, h2) in enumerate(layer_sizes):
            classifier.add_module('fc'+str(i+1), nn.Linear(h1, h2))
            classifier.add_module('relu'+str(i+1), nn.ReLU())
            classifier.add_module('drop'+str(i+1), nn.Dropout(.5))
        classifier.add_module('output', nn.Linear(hidden_layers[-1], num_out_features))
        
    return classifier


def train_model(model, criterion, optimizer, sched, num_epochs=5):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        #sched.step()
                        loss.backward()
                        
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            # deep copy the model
            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    #load best model weights
    model.load_state_dict(best_model_wts)
    
    return model


def predict(model, dataloaders):
    model.eval()
    predlist = torch.zeros(0, dtype=torch.long, device='cpu')
    lbllist = torch.zeros(0, dtype=torch.long, device='cpu')

    for inputs, labels in dataloaders['test']:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        # Class with the highest probability is our predicted class     
        predlist = torch.cat([predlist,outputs.max(1)[1].view(-1).cpu()])
        lbllist = torch.cat([lbllist,labels.view(-1).cpu()])
        
    return lbllist, predlist


def calculate_metrics(y_test, pred):
    cm = confusion_matrix(y_test, pred)
    cmn = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # normalise
    acc = accuracy_score(y_test, pred)
    precision = precision_score(y_test, pred, average=None)
    recall = recall_score(y_test, pred, average=None)
    f_score = f1_score(y_test, pred, average=None)
    return acc, precision, recall, f_score, cm, cmn


def print_metrics(acc, precision, recall, f_score):
    print(f'Accuracy:  {acc:.2f}\nPrecision: {precision.mean():.2f} {precision}\n\
      Recall:    {recall.mean():.2f} {recall}\nF1_score:  {f_score.mean():.2f} {f_score}')
    
    
def show_confusion_matrix(cm, classes):
    df_cm = pd.DataFrame(cm, classes, classes)
    sns.heatmap(df_cm, annot=True)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.title('Confusion Matrix')
    plt.show()
    
    
if __name__ == '__main__':
    device = get_device()
    print(device)
    
    image_datasets, dataloaders, class_names, dataset_sizes = prepare_datasets(part=True)
    # images, labels = next(iter(dataloaders['train']))
    
    model_name = 'densenet'
    model, num_in_features = create_model('densenet')

    hidden_layers = None #[4096, 1024, 256][512, 256, 128]
    classifier = create_classifier(num_in_features, hidden_layers, 102)
    criterion, optimizer, sched = bind_model_classifier(model_name, classifier)
    
    epochs = 30
    model.to(device)
    model = train_model(model, criterion, optimizer, sched, epochs)

    lbllist, predlist = predict(model, dataloaders)

    acc, precision, recall, f_score, cm, cmn = calculate_metrics(lbllist.numpy(), predlist.numpy())
    print_metrics(acc, precision, recall, f_score)
    show_confusion_matrix(cmn, image_datasets['test'].classes)