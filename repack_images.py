import os
import shutil
from tqdm import tqdm

'''
The images from the ImageNet website are not sorted into folders by class
This script does that for the 2010 validation data
'''

filenames = os.listdir('val')
filenames.sort()
val_labels = open('/home/asemota_alexander/data/ILSVRC2010_val_ground_truth.txt', 'r').readlines()

print(min(val_labels))
for i in range(1000):
    if not os.path.exists(os.path.join('/home/asemota_alexander/data/val_2010', str(i))):
        os.makedirs(os.path.join('/home/asemota_alexander/data/val_2010',str(i)))

for i in tqdm(range(len(filenames))):
    label = int(val_labels[i])
    shutil.move(os.path.join('/home/asemota_alexander/project/val',filenames[i]), 
            os.path.join('/home/asemota_alexander/data/val_2010',str(label)))

'''
There seem to only be classes 1-999, instead of 0-999
I'm not sure if this means I'm missing something or not
'''
