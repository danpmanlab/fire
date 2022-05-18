
import json
import csv
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
import pickle
from cirtorch.datasets.datahelpers import default_loader
from curses import keyname
import sys
import argparse
from pathlib import Path
import yaml 
import ast
import time
import torch
from torchvision import transforms

from how.utils import io_helpers, logging, download
from how.stages.evaluate import eval_asmk
# from examples.demo_how import DATASET_URL
import fire_network

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

def caculate_blur(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm

def save_img_2(query,results,save_folder,id):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    fig, axis = plt.subplots(1,1+len(results), figsize=(20,5))
    query_img = np.array(query)
    axis[0].imshow(query_img)
    axis[0].set_axis_off()
    for i,path in enumerate(results):
        rt_img = Image.open("/media/anlab/DATA/lashinbang-server/revamptracking/output/5350_new/data_full/"+path)
        rt_img = np.array(rt_img)
        axis[i+1].imshow(rt_img)
        axis[i+1].set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(save_folder,id+".jpg"))
    plt.close()
def save_img(images_list,save_folder):
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    query_path = images_list[0]

    results = images_list[1:]
    fig, axis = plt.subplots(1,len(images_list), figsize=(20,5))
    query_img = Image.open(query_path)
    query_img = np.array(query_img)
    axis[0].imshow(query_img)
    axis[0].set_axis_off()
    name_file = query_path.split('/')[-1]

    for i,path in enumerate(results):
        rt_img = Image.open(path)
        rt_img = np.array(rt_img)
        # if i == 0:
        #     rt_img = cv2.rectangle(rt_img, (x,y),(xmax,ymax),(0,255,0),5)
        axis[i+1].imshow(rt_img)
        axis[i+1].set_axis_off()
    fig.tight_layout()
    fig.savefig(os.path.join(save_folder,name_file))
    plt.close()

def resize(img,size):
    height,width= img.shape[:2]
    ratio= size/max(height,width)
    width_new= int(width*ratio)
    height_new=int(height*ratio)
    img=cv2.resize(img,(width_new,height_new))
    return img



def im_resize(img, imsize):
    w, h = img.size
    if h > w:
        new_h = imsize
        new_w = int(w * new_h/h)
    else:
        new_w = imsize
        new_h = int(h * new_w / w)
    img = img.resize((int(new_w), int(new_h)))
    return img
class Resize_ratio():
    def __init__(self, imsize):
        self.imsize = imsize
    def __call__(self, image):
        image = im_resize(image, self.imsize)
        return image
DATASET_URL = "http://ptak.felk.cvut.cz/personal/toliageo/share/how/dataset/"
def _overwrite_cirtorch_path(root_path):
    """Hack to fix cirtorch paths"""
    from cirtorch.datasets import traindataset
    from cirtorch.networks import imageretrievalnet

    traindataset.get_data_root = lambda: root_path
    imageretrievalnet.get_data_root = lambda: root_path

def evaluate_demo(demo_eval, evaluation, globals):
    globals["device"] = torch.device("cpu")
    if demo_eval['gpu_id'] is not None:
        globals["device"] = torch.device(("cuda:%s" % demo_eval['gpu_id']))

    # Handle net_path when directory
    net_path = Path(demo_eval['exp_folder']) / demo_eval['net_path']
    if net_path.is_dir() and (net_path / "epochs/model_best.pth").exists():
        net_path = net_path / "epochs/model_best.pth"

    # Load net
    state = torch.load(net_path, map_location='cpu')
    state['net_params']['pretrained'] = None # no need for imagenet pretrained model
    net = fire_network.init_network(**state['net_params']).to(globals['device'])
    net.load_state_dict(state['state_dict'])
    globals["transform"] = transforms.Compose([
                Resize_ratio(500),
                transforms.ToTensor(), \
                transforms.Normalize(**dict(zip(["mean", "std"], net.runtime['mean_std'])))])

    # Eval
    
    # evaluation['inference']=None
    print("evaluation['inference']",evaluation['inference'])
    eval_asmk(net, evaluation['inference'], globals, **evaluation['local_descriptor'])