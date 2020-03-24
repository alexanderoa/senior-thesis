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
transform = transforms.Compose([transforms.Resize(256),
                                transforms.CenterCrop(224),
                                transforms.ToTensor(), 
                                normalize])
imagenetv2 = datasets.ImageFolder(root='imagenetv2-matched-frequency', transform=transform)
imagenetv1 = datasets.ImageFolder(root='/home/asemota_alexander/data/val_2010', transform=transform) #using val for now, for efficiency
testv1 = torch.utils.data.DataLoader(imagenetv1, batch_size=4,
                                        shuffle=True, num_workers=1)
testv2 = torch.utils.data.DataLoader(imagenetv2, batch_size=4,
                                        shuffle=True, num_workers=1)


len_classes = 1000
index_to_cat = [str(x) for x in range(len_classes)]
index_to_cat.sort()
index_to_cat = torch.tensor([int(x) for x in index_to_cat])

def num_correct(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    #yanked from https://github.com/bearpaw/pytorch-classification/blob/cc9106d598ff1fe375cc030873ceacfea0499d77/utils/eval.py
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    pred = pred.cpu()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k)
    return res

def run_models(models, testloader):
    results = np.zeros((len(models),2))
    for k in tqdm(range(len(models))):
        correct = 0
        total = 0
        model = models[k][0]
        model.to(device)
        model = model.eval()
        testdata = iter(testloader)
        for j in range(len(testdata)):
            images, labels = testdata.next()
            images = images.to(device)
            outputs = model(images)
            right_labels = index_to_cat[labels]
            correct = num_correct(outputs, right_labels, (1,5)) #get top-1 and top-5
            total += labels.size(0)
            results[k][0] = correct[0]/total #top-1 accuracy
            results[k][1] = correct[1]/total #top-5 accuracy
        torch.cuda.empty_cache()
    return results

model_acc = np.zeros((2, len(model_list), 2))
print('Running pretrained models on ImageNetV1...')
model_acc[0] = run_models(model_list, testv1)
print('Running pretrained models on ImageNetV2...')
model_acc[1] = run_models(model_list, testv2)
np.save('model_acc', model_acc)

torch_acc = np.zeros((2, len(torch_list), 2))
print('Running torch models on ImageNetV1...')
torch_acc = run_models(torch_list, testv1)

print('Running torch models on ImageNextV2...')

torch_acc = run_models(torch_list, testv2)
np.save('torch_acc', torch_acc)


