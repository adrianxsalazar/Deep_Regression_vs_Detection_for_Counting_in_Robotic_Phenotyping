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
import matplotlib.image as mpimg


parser = argparse.ArgumentParser(description='Test PyTorch CSRNet')

parser.add_argument('test_json', metavar='test',
                    help='path to val json')

parser.add_argument('output', metavar='VAL',
                    help='path output')

args = parser.parse_args()


transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),])



def save_dictionary(dictpath_json, dictionary_data):
    a_file = open(dictpath_json, "w")
    json.dump(dictionary_data, a_file)
    a_file.close()

#Select the image extension

with open(args.test_json, 'r') as outfile:
    img_paths = json.load(outfile)

#
model = CSRNet()
pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(pytorch_total_params)

#defining the model
model = model.cuda()

#loading the trained weights
checkpoint = torch.load(os.path.join(args.output,'model_best.pth.tar'))
#load best model
model.load_state_dict(checkpoint['state_dict'])

mae = 0
pred= []
gt = []

dictionary_counts={}

for i in xrange(len(img_paths)):
    plain_file=os.path.basename(img_paths[i])
    #print (plain_file)
    #open image
    img = transform(Image.open(img_paths[i]).convert('RGB')).cuda()
    gt_file = h5py.File(img_paths[i].replace('.jpg','.h5'))
    groundtruth = np.asarray(gt_file['density'])

    output = model(img.unsqueeze(0))
    
    dictionary_counts[plain_file]=round(abs(output.detach().cpu().sum().numpy()),3)   

    mae += abs(output.detach().cpu().sum().numpy()-np.sum(groundtruth))

    pred.append(output.detach().cpu().sum().numpy())
    gt.append(np.sum(groundtruth))
    print (output.detach().cpu().sum().numpy(),np.sum(groundtruth) )
    
    
    
    ###################################################################
    
    #print("Predicted Count : ",int(output.detach().cpu().sum().numpy()))
    temp = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[2],output.detach().cpu().shape[3]))
    plt.imshow(temp,cmap = c.jet)
    plt.axis('off')
    
    #change the name of the image
    plt.savefig(os.path.join(args.output,'visual_results',plain_file), bbox_inches='tight', pad_inches = 0)
    plt.close()

    plt.imshow(groundtruth,cmap = c.jet)
    plt.axis('off')
    plt.savefig(os.path.join(args.output,'visual_results','gt_original'+plain_file),bbox_inches='tight',pad_inches = 0)
    plt.close()
    
    plt.imshow(mpimg.imread(img_paths[i]))
    plt.axis('off')
    plt.savefig(os.path.join(args.output,'visual_results','original'+plain_file),bbox_inches='tight',pad_inches = 0)
    plt.close()

    #######################################################################

print (mae/len(img_paths))

mae_I = mean_absolute_error(pred,gt)
rmse = np.sqrt(mean_squared_error(pred,gt))

save_dictionary(os.path.join(args.output,"dic_restults.json"),dictionary_counts)
print(os.path.join(args.output,"dic_restults.json"))

print ('MAE: ',mae)
print ('MAE I', mae_I)
print ('RMSE: ',rmse)
results=np.array([mae,mae_I,rmse])
np.savetxt(os.path.join(args.output,"restults.txt"),results,delimiter=',')
