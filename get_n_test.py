import pretrainedmodels
import numpy as np
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets

model_names = pretrainedmodels.model_names
model_list = []
no_load = []

print('Downloading models...')
for i in tqdm(range(len(model_names))):
    try:
        model_list.append((pretrainedmodels.__dict__[model_names[i]](num_classes=1000, pretrained='imagenet'), 
                            model_names[i]))
    except:
        print("Couldn't download", model_names[i])
        no_load.append(model_names[i])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Moving models to GPU...')
for i in tqdm(range(len(model_list))):
    model = model_list[i][0]
    model.to(device)
    model.eval() #put all models in evaluation mode


normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]) #normalize input
transform = transforms.Compose([transforms.Resize(300),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(), 
                                normalize])
incept = transforms.Compose([transforms.Resize(400),
                                transforms.CenterCrop(299),
                                transforms.ToTensor(),
                                normalize])

imagenetv2 = datasets.ImageFolder(root='imagenetv2-matched-frequency', transform=transform) 
testloader = torch.utils.data.DataLoader(imagenetv2, batch_size=4,
                                        shuffle=True, num_workers=1)
inceptnetv2 = datasets.ImageFolder(root='imagenetv2-matched-frequency', transform=incept)
inceptloader = torch.utils.data.DataLoader(imagenetv2, batch_size=4,
                                            shuffle=True, num_workers=1)
not_run = []
print('Testing models...')
count = 0
testdata = iter(testloader)
inceptdata = iter(inceptloader)
image1, label1 = testdata.next()
image2, label2 = inceptdata.next()
image1 = image1.to(device)
image2 = image2.to(device)
for k in tqdm(range(len(model_list))):
    model = model_list[k][0]
    try:
        outputs = model(image1)
    except:
        print('Regular loader didnt work. Trying incept loader')
    else:
        continue

    try:
        outputs = model(image2)
    except:
        print('Incept data didnt work for', model_list[k][1])
        not_run.append(model_list[k][1])
    '''
    images.to('cpu')
    labels.to('cpu')
    '''

print(len(not_run), 'models didnt run')

np.save('not_run', not_run)
np.save('no_load', no_load)
