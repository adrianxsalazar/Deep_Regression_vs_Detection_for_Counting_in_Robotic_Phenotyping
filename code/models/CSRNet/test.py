#text and inference for the CSRNet model
import h5py
import scipy.io as io
import PIL.Image as Image
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter
import scipy
import json
from sklearn.metrics import mean_squared_error,mean_absolute_error
import torchvision.transforms.functional as F
from matplotlib import cm as CM
from image import *
from model import CSRNet
import torch
import argparse
from torchvision import datasets, transforms
from matplotlib import cm as c
import tqdm


parser = argparse.ArgumentParser(description='Test PyTorch CSRNet')

parser.add_argument('test_json', metavar='test',
                    help='path to val json')

parser.add_argument('output', metavar='VAL',
                    help='path output')

args = parser.parse_args()



transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),])


#Select the image extension

with open(args.test_json, 'r') as outfile:
    img_paths = json.load(outfile)

#
model = CSRNet()

#defining the model
model = model.cuda()

#loading the trained weights
checkpoint = torch.load(os.path.join(args.output,'model_best.pth.tar'))
#load best model
model.load_state_dict(checkpoint['state_dict'])

mae = 0
pred= []
gt = []

for i in xrange(len(img_paths)):
    #open image
    img = transform(Image.open(img_paths[i]).convert('RGB')).cuda()
    gt_file = h5py.File(img_paths[i].replace('.png','.h5'))
    groundtruth = np.asarray(gt_file['density'])

    output = model(img.unsqueeze(0))

    mae += abs(output.detach().cpu().sum().numpy()-np.sum(groundtruth))

    pred.append(output.detach().cpu().sum().numpy())
    gt.append(np.sum(groundtruth))
    print (output.detach().cpu().sum().numpy(),np.sum(groundtruth) )

print (mae/len(img_paths))

mae_I = mean_absolute_error(pred,gt)
rmse = np.sqrt(mean_squared_error(pred,gt))

print ('MAE: ',mae)
print ('MAE I', mae_I)
print ('RMSE: ',rmse)
results=np.array([mae,mae_I,rmse])
np.savetxt(os.path.join(args.output,"restults.txt"),results,delimiter=',')
