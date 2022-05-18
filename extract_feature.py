
# import sys
# sys.path.append("/media/anlab/data/lashinbang/lashinbang-server/fire/FIRe/asmk")
# sys.path.append("/media/anlab/data/lashinbang/lashinbang-server/fire/FIRe/cnnimageretrieval-pytorch-1.2")
# sys.path.append("/media/anlab/data/lashinbang/lashinbang-server/fire/FIRe/how")
# sys.path.append("/media/anlab/data/lashinbang/lashinbang-server/fire/FIRe")
# sys.path.append("fire")
import pickle
from cirtorch.datasets.genericdataset import ImagesFromList
import json
import csv
import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from cirtorch.datasets.datahelpers import default_loader
from curses import keyname
from pathlib import Path
import time
import torch
from torchvision import transforms
from how.utils import io_helpers, logging, download
from how.stages.evaluate import eval_asmk,asmk_train_codebook
from how.networks.how_net import extract_vectors_local
import fire_network
from util_evaluate import Resize_ratio,caculate_blur,_overwrite_cirtorch_path,DATASET_URL,save_img,variance_of_laplacian,save_img_2
from asmk import asmk_method,kernel as kern_pkg
# from FIRe.evaluate import asmk_train_codebook
def get_root(path_file):
    # print("path_file",path_file)
    value= path_file.split("/")[:-1]
    folder_save= ""
    for item in value:
        folder_save=folder_save+f"/{item}"
    return folder_save

def extract(path_base,list_path,path_foder_save):
    
    start_time = time.time()
    """Argument parsing and parameter preparation for the demo"""
    parameters=str(Path(__file__).resolve().parent)+'/eval_fire.yml'
    args={
        "experiment":None,
        "demo_eval.net_path":None,
        "evaluation.inference.features_num":None,
        "evaluation.inference.scales":None
    }
    # Load yaml params
    package_root = Path(__file__).resolve().parent
    # parameters_path = args.parameters
    parameters_path=parameters
    params = io_helpers.load_params(parameters_path)
    # Overlay with command-line arguments
    for key in args.keys():
        arg=key
        val=args[key]
        # print("arg,val",arg,val)
        if arg not in {"command", "parameters"} and val is not None:
            io_helpers.dict_deep_set(params, arg.split("."), val)
            
    # Resolve experiment name
    exp_name = params.pop("experiment")
    if not exp_name:
        exp_name = Path(parameters_path).name[:-len(".yml")]

    # Resolve data folders
    globals = {}
    globals["root_path"] = (package_root / params['demo_eval']['data_folder'])
    globals["root_path"].mkdir(parents=True, exist_ok=True)
    _overwrite_cirtorch_path(str(globals['root_path']))
    globals["exp_path"] = (package_root / params['demo_eval']['exp_folder']) / exp_name
    globals["exp_path"].mkdir(parents=True, exist_ok=True)
    # Setup logging
    globals["logger"] = logging.init_logger(globals["exp_path"] / f"eval.log")

    # Run demo
    io_helpers.save_params(globals["exp_path"] / f"eval_params.yml", params)
    params['evaluation']['global_descriptor'] = dict(datasets=[])

    download.download_for_eval(params['evaluation'], params['demo_eval'], DATASET_URL, globals)
    # evaluate_demo(**params, globals=globals)


    globals["device"] = torch.device("cpu")
    if params['demo_eval']['gpu_id'] is not None:
        globals["device"] = torch.device(("cuda:%s" % params['demo_eval']['gpu_id']))

    # Handle net_path when directory
    net_path = Path(params['demo_eval']['exp_folder']) / params['demo_eval']['net_path']
    if net_path.is_dir() and (net_path / "epochs/model_best.pth").exists():
        net_path = net_path / "epochs/model_best.pth"
    print("--- time before load net ---"  ,(time.time()- start_time))
    # Load net
    state = torch.load(net_path, map_location='cpu')
    state['net_params']['pretrained'] = None # no need for imagenet pretrained model
    net = fire_network.init_network(**state['net_params']).to(globals['device'])
    net.load_state_dict(state['state_dict'])
    globals["transform"] = transforms.Compose([
                Resize_ratio(500),
                transforms.ToTensor(), \
                transforms.Normalize(**dict(zip(["mean", "std"], net.runtime['mean_std'])))])

    time_before_eval = time.time()
    print("--- time before eval ---"  ,(time_before_eval - start_time))
    # Eval
    #############method 2###############################################################

    # path_data="/media/anlab/data/lashinbang/lashinbang-server/revamptracking/output/5337_new/data_full/"
    # data=os.listdir(path_data)
    # data=[path_data+value for value in data]
    # print("len",len(data))

    print("globals",globals)

    asmk=params['evaluation']['local_descriptor']['asmk']
    asmk = asmk_method.ASMKMethod.initialize_untrained(asmk)
    asmk = asmk_train_codebook(net, params['evaluation']['inference'], globals, globals["logger"], codebook_training=params['evaluation']['local_descriptor']['codebook_training'],
                                asmk=asmk, cache_path=globals["exp_path"] / "asmk.pkl")


    data_opts = {"imsize": params['evaluation']['inference']['image_size'], "transform": globals['transform']}
    infer_opts = {"scales": params['evaluation']['inference']['scales'], "features_num": params['evaluation']['inference']['features_num']}
    
    data=[os.path.join(path_base,value) for value in list_path]
    dset = ImagesFromList(root='', images=data, bbxs=None, **data_opts)

    # folder_path="/media/anlab/data/lashinbang/lashinbang-server/revamptracking/output/5337_new/pickle_full/"
    
    vecs, imids, *_ = extract_vectors_local(net, dset, globals["device"], **infer_opts)
    for i,value in enumerate(data):
        name= "/"+value.split("/")[-1]
        # print("path_foder_save",path_foder_save)
        # foder_create= os.path.join(path_foder_save,get_root(list_path[i]))

        foder_create=path_foder_save+get_root(list_path[i])

        # print("foder_create",foder_create)
        if(os.path.exists(foder_create)==False):
            os.makedirs(foder_create)
        with open(foder_create+name+".pickle","wb") as f:
            pickle.dump(vecs[i*125:i*125+125],f)
            
