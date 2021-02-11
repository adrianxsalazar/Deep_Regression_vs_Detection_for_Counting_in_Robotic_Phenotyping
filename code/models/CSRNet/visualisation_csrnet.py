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
import torchvision.transforms.functional as F
from matplotlib import cm as CM
from image import *
from model import CSRNet
import torch
import argparse

from torchvision import datasets, transforms
from matplotlib import cm as c


parser = argparse.ArgumentParser(description='Test PyTorch CSRNet')

parser.add_argument('path_testing_image', metavar='Test_image_path',
                    help='path to train json')

parser.add_argument('best_model_csrnet_path', metavar='TEST',
                    help='path to test json')
args = parser.parse_args()

def main():
    global args

    print (args.path_testing_image)

    transform=transforms.Compose([transforms.ToTensor(),transforms.Normalize(mean=[0.485, 0.456, 0.406],
                            std=[0.229, 0.224, 0.225]),])

    img_paths=[]
    for img_path in glob.glob(os.path.join(args.path_testing_image, '*.png')):
        img_paths.append(img_path)

    model = CSRNet()

    #defining the model
    model = model.cuda()

    #loading the trained weights
    checkpoint = torch.load(args.best_model_csrnet_path)
    model.load_state_dict(checkpoint['state_dict'])

    img = transform(Image.open(args.path_testing_image).convert('RGB')).cuda()
    output = model(img.unsqueeze(0))
    print("Predicted Count : ",int(output.detach().cpu().sum().numpy()))
    temp = np.asarray(output.detach().cpu().reshape(output.detach().cpu().shape[2],output.detach().cpu().shape[3]))
    plt.imshow(temp,cmap = c.jet)
    plt.axis('off')
    plt.savefig("predicted_dt.png",bbox_inches='tight')

    temp = h5py.File(args.path_testing_image.replace('.png','.h5'))
    temp_1 = np.asarray(temp['density'])
    plt.imshow(temp_1,cmap = c.jet)
    print(" Original Count : ",int(np.sum(temp_1)) + 1)
    plt.axis('off')
    plt.savefig("original_dt.png",bbox_inches='tight')

    print("Original Image")
    plt.imshow(plt.imread(args.path_testing_image))
    plt.axis('off')
    plt.savefig("original_image.png",bbox_inches='tight')

    f = open('results.txt','w')
    f.write("Predicted Count : " + str(int(output.detach().cpu().sum().numpy()))+ " Original Count : " + str(int(np.sum(temp_1)) + 1))
    f.close()

if __name__ == '__main__':
    main()
