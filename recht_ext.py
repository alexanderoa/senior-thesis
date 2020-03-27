import pretrainedmodels
import sys
import numpy as np
from tqdm import tqdm
import torch
import torchvision
import torchvision.transforms as transforms
from torchvision import datasets
import torchvision.models as models

model_names = pretrainedmodels.model_names
model_list = []
no_load = np.load('no_load.npy') #list of models from pretrainedmodels that don't load or run data properly
not_run = np.load('not_run.npy')
test_order = []
forget = list(no_load) + list(not_run)

print('Downloading models...')
for i in tqdm(range(len(model_names))): #retrieving models from pretrainemodels package
    if model_names[i] in forget:
        continue
    model_list.append((pretrainedmodels.__dict__[model_names[i]](num_classes=1000, pretrained='imagenet'), 
                            model_names[i]))
    test_order.append(model_names[i])

'''
loading pretrainemodels from torchvision
there's overlap with the pretrainemodels package
'''
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
def rotate_imgs(dataloc, rotation=(0,0)):
    rotation = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.RandomRotation(degrees=rotation),
                                    transforms.ToTensor(), 
                                    normalize])
    img_rot = datasets.ImageFolder(root=dataloc, transform=rotation)
    test_rot = torch.utils.data.DataLoader(img_rot, batch_size=4,
                                            shuffle=True, num_workers=1)
    return test_rot

def add_noise(dataloc, noise=0, mean=0.45, std=0.224, mean_mtplr=1, std_mtplr=1):
    noisy_transform = transforms.Compose([transforms.Resize(256),
                                    transforms.CenterCrop(224),
                                    transforms.ToTensor(), 
                                    transforms.Lambda(lambda x : x + 
                                        noise*torch.empty_like(x).normal_(mean=mean_mtplr*mean, 
                                            std=std_mtplr*std)),
                                    normalize])
    img_noise = datasets.ImageFolder(root=dataloc, transform=noisy_transform)
    test_noise = torch.utils.data.DataLoader(img_noise, batch_size=4,
                                            shuffle=True, num_workers=1)
    return test_noise

imagenetv2 = datasets.ImageFolder(root='imagenetv2-matched-frequency', transform=transform)
testv2 = torch.utils.data.DataLoader(imagenetv2, batch_size=4,
                      shuffle=True, num_workers=1)
v2_folder = 'imagenetv2-matched-frequency'
rotate90 = rotate_imgs(v2_folder, (90,90))
rotate180 = rotate_imgs(v2_folder, (180, 180))
rotate45 = rotate_imgs(v2_folder, (45, 45))
rotate_ran = rotate_imgs(v2_folder, (0, 360))

noise10 = add_noise(v2_folder, noise=0.1)
noise25 = add_noise(v2_folder, noise=0.25)
noise50 = add_noise(v2_folder, noise=0.5)

wide_noise10 = add_noise(v2_folder, noise=0.1, std_mtplr=2)
wide_noise25 = add_noise(v2_folder, noise=0.25, std_mtplr=2)
wide_noise50 = add_noise(v2_folder, noise=0.5, std_mtplr=2)

nrw_noise10 = add_noise(v2_folder, noise=0.1, std_mtplr=0.5)
nrw_noise25 = add_noise(v2_folder, noise=0.25, std_mtplr=0.5)
nrw_noise50 = add_noise(v2_folder, noise=0.5, std_mtplr=0.5)

'''
DataLoader loads the models in order by digit (e.g. 1, 10, 11, ...) rather than numerical order
index_to_cat let's us convert the labels from DataLoader to the correct labels
'''
len_classes = 1000
index_to_cat = [str(x) for x in range(len_classes)]
index_to_cat.sort()
index_to_cat = torch.tensor([int(x) for x in index_to_cat])

def num_correct(output, target, topk=(1,)):
    '''
    Computes the precision@k for the specified values of k
    Yanked from github:
    https://github.com/bearpaw/pytorch-classification/blob/cc9106d598ff1fe375cc030873ceacfea0499d77/utils/eval.py
    '''
    maxk = max(topk)
    batch_size = target.size(0)
    size = output.size()
    if len(size)>2:
        output = output.view((size[0], size[1]))
    assert(output.size(0) == target.size(0))
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
    '''
    Function to run a list of models on ImageNet or a transform of ImageNet
    Calculates top-1 and top-5 accuracy
    Loads models and images onto the GPU one-by-one to avoid memory issues
    Empties GPU cache after testing each model
    '''
    results = np.zeros((len(models),2))
    for k in tqdm(range(len(models))):
        correct = [0,0]
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
            between = num_correct(outputs, right_labels, (1,5)) #get top-1 and top-5
            correct[0] += between[0]
            correct[1] += between[1]
            total += labels.size(0)
        results[k][0] = correct[0]/total #top-1 accuracy
        results[k][1] = correct[1]/total #top-5 accuracy
        torch.cuda.empty_cache()
    return results
'''
testv2_acc = np.zeros((len(model_list)+len(torch_list), 2))
print('Testing models on ImageNetV2...')
testv2_acc[:len(model_list)] = run_models(model_list, testv2)
testv2_acc[len(model_list):] = run_models(torch_list, testv2)
np.save('results/testv2_acc', testv2_acc)

rot180_acc = np.zeros((len(model_list)+len(torch_list), 2))
print('Testing models on upside down images...')
rot180_acc[:len(model_list)] = run_models(model_list, rotate180)
rot180_acc[len(model_list):] = run_models(torch_list, rotate180)
np.save('results/rot180_acc', rot180_acc)

rot90_acc = np.zeros((len(model_list)+len(torch_list), 2))
print('Testing models on images rotated 90 deg...')
rot90_acc[:len(model_list)] = run_models(model_list, rotate90)
rot90_acc[len(model_list):] = run_models(torch_list, rotate90)
np.save('results/rot90_acc', rot90_acc)

rot45_acc = np.zeros((len(model_list)+len(torch_list), 2))
print('Testing models on images rotated 45 deg...')
rot45_acc[:len(model_list)] = run_models(model_list, rotate45)
rot45_acc[len(model_list):] = run_models(torch_list, rotate45)
np.save('results/rot45_acc', rot45_acc)

rotran_acc = np.zeros((len(model_list)+len(torch_list), 2))
print('Testing models on images rotated randomly...')
rotran_acc[:len(model_list)] = run_models(model_list, rotate_ran)
rotran_acc[len(model_list):] = run_models(torch_list, rotate_ran)
np.save('results/rotran_acc', rotran_acc)
'''

noise10_acc = np.zeros((len(model_list)+len(torch_list), 2))
print('Testing models on images plus ten percent noise...')
noise10_acc[:len(model_list)] = run_models(model_list, noise10)
noise10_acc[len(model_list):] = run_models(torch_list, noise10)
np.save('results/noise10_acc', noise10_acc)

noise25_acc = np.zeros((len(model_list)+len(torch_list), 2))
print('Testing models on images plus 25 percent noise...')
noise25_acc[:len(model_list)] = run_models(model_list, noise25)
noise25_acc[len(model_list):] = run_models(torch_list, noise25)
np.save('results/noise25_acc', noise25_acc)

noise50_acc = np.zeros((len(model_list)+len(torch_list), 2))
print('Testing models on images plus 50 percent noise...')
noise50_acc[:len(model_list)] = run_models(model_list, noise50)
noise50_acc[len(model_list):] = run_models(torch_list, noise50)
np.save('results/noise50_acc', noise50_acc)

wide_noise10_acc = np.zeros((len(model_list)+len(torch_list), 2))
print('Testing models on images plus 10% wide noise...')
wide_noise10_acc[:len(model_list)] = run_models(model_list, wide_noise10)
wide_noise10_acc[len(model_list):] = run_models(torch_list, wide_noise10)
np.save('results/wide_noise10_acc', wide_noise10_acc)

wide_noise25_acc = np.zeros((len(model_list)+len(torch_list), 2))
print('Testing models on images plus 25% wide noise...')
wide_noise25_acc[:len(model_list)] = run_models(model_list, wide_noise25)
wide_noise25_acc[len(model_list):] = run_models(torch_list, wide_noise25)
np.save('results/wide_noise25_acc', wide_noise25_acc)

wide_noise50_acc = np.zeros((len(model_list)+len(torch_list), 2))
print('Testing models on images plus 50% wide noise...')
wide_noise50_acc[:len(model_list)] = run_models(model_list, wide_noise50)
wide_noise50_acc[len(model_list):] = run_models(torch_list, wide_noise50)
np.save('results/wide_noise50_acc', wide_noise50_acc)

nrw_noise10_acc = np.zeros((len(model_list)+len(torch_list), 2))
print('Testing models on images plus 10% narrow noise...')
nrw_noise10_acc[:len(model_list)] = run_models(model_list, nrw_noise10)
nrw_noise10_acc[len(model_list):] = run_models(torch_list, nrw_noise10)
np.save('results/nrw_noise10_acc', nrw_noise10_acc)

nrw_noise25_acc = np.zeros((len(model_list)+len(torch_list), 2))
print('Testing models on images plus 25% narrow noise...')
nrw_noise25_acc[:len(model_list)] = run_models(model_list, nrw_noise25)
nrw_noise25_acc[len(model_list):] = run_models(torch_list, nrw_noise25)
np.save('results/nrw_noise25_acc', nrw_noise25_acc)

nrw_noise50_acc = np.zeros((len(model_list)+len(torch_list), 2))
print('Testing models on images plus 50% narrow noise...')
nrw_noise50_acc[:len(model_list)] = run_models(model_list, nrw_noise50)
nrw_noise50_acc[len(model_list):] = run_models(torch_list, nrw_noise50)
np.save('results/nrw_noise50_acc', nrw_noise50_acc)

