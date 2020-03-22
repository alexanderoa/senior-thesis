import pretrainedmodels
import numpy as np
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import torchvision.models as models

model_names = pretrainedmodels.model_names
model_list = []
no_load = np.load('no_load.npy')
not_run = np.load('not_run.npy')
test_order = []
forget = list(no_load) + list(not_run)

print('Downloading models...')
for i in tqdm(range(len(model_names))):
    if model_names[i] in forget:
        continue
    model_list.append((pretrainedmodels.__dict__[model_names[i]](num_classes=1000, pretrained='imagenet'), 
                            model_names[i]))
    test_order.append(model_names[i])

resnet18 = models.resnet18(pretrained=True)
alexnet = models.alexnet(pretrained=True)
squeezenet = models.squeezenet1_0(pretrained=True)
vgg16 = models.vgg16(pretrained=True)
densenet = models.densenet161(pretrained=True)
googlenet = models.googlenet(pretrained=True)
shufflenet = models.shufflenet_v2_x1_0(pretrained=True)
mobilenet = models.mobilenet_v2(pretrained=True)
resnext50_32x4d = models.resnext50_32x4d(pretrained=True)
wide_resnet50_2 = models.wide_resnet50_2(pretrained=True)
mnasnet = models.mnasnet1_0(pretrained=True)
torch_list = [resnet18, alexnet, squeezenet, vgg16, densenet, googlenet, 
        shufflenet, mobilenet, resnext50_32x4d, wide_resnet50_2, mnasnet]
torch_names = ['resnet18', 'alexnet', 'squeezenet', 'vgg16', 'densenet', 'googlenet', 
        'shufflenet', 'mobilenet', 'resnext50_32x4d', 'wide_resnet50_2', 'mnasnet']
for i in range(len(torch_list)):
    torch_list[i] = (torch_list[i], torch_names[i])
    test_order.append(torch_names[i])

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225]) #normalize input
transform = transforms.Compose([transforms.Resize(300),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(), 
                                normalize])
imagenetv2 = datasets.ImageFolder(root='imagenetv2-matched-frequency', transform=transform) 
testloader = torch.utils.data.DataLoader(imagenetv2, batch_size=4,
                                        shuffle=True, num_workers=1)

len_classes = 1000
index_to_cat = [str(x) for x in range(len_classes)]
index_to_cat.sort()
index_to_cat = torch.tensor([int(x) for x in index_to_cat])

print('Testing models...')
model_acc = np.zeros(len(model_list))
torch_acc = np.zeros(len(torch_list))

print('Testing pretrainedmodels')
for k in tqdm(range(len(model_list))):
    correct = 0
    total = 0
    model = model_list[k][0]
    model.to(device)
    model = model.eval()
    testdata = iter(testloader)
    for j in range(len(testdata)):
        images, labels = testdata.next()
        images = images.to(device)
        outputs = model(images)
        _, pred = torch.max(outputs.data, 1)
        right_labels = index_to_cat[labels]
        total += labels.size(0)
        correct += (right_labels == pred.cpu()).sum().item()
        model_acc[k] = correct/total
    torch.cuda.empty_cache()
np.save('model_acc', model_acc)

model_acc = np.load('model_acc.npy')
print(model_acc)
print('Test torch models')
for k in tqdm(range(len(torch_list))):
    correct = 0
    total = 0
    model = torch_list[k][0]
    model.to(device)
    model = model.eval()
    testdata = iter(testloader)
    for j in range(len(testdata)):
        images, labels = testdata.next()
        images = images.to(device)
        outputs = model(images)
        _, pred = torch.max(outputs.data, 1)
        right_labels = index_to_cat[labels]
        total += labels.size(0)
        correct += (right_labels == pred.cpu()).sum().item()
        torch_acc[k] = correct/total
    torch.cuda.empty_cache()
np.save('torch_acc', torch_acc)
