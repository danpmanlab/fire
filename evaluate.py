# Copyright (C) 2021-2022 Naver Corporation. All right0 reserved.
# Licensed under CC BY-NC-SA 4.0 (non-commercial use only).
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
from how.stages.evaluate import eval_asmk
import fire_network
from util_evaluate import Resize_ratio,caculate_blur,_overwrite_cirtorch_path,DATASET_URL,save_img,variance_of_laplacian,save_img_2
from PIL import Image, ImageOps
def run(img_represent_new,data,ids,path_foder_save):
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
    print("params",params.keys())
    print("params",params)
    ranks=eval_asmk(path_foder_save,img_represent_new,data,net, params['evaluation']['inference'], globals, **params['evaluation']['local_descriptor'])[:,:5]
    # dict_result= eval_asmk(img_represent_new,data,net, params['evaluation']['inference'], globals, **params['evaluation']['local_descriptor'])[:,:5]
    # ranks=dict_result["ranks"]
    # print("ranks",ranks.shape)
    # print("scores",scores[0])
    data=np.array(data)
    image_correspond=np.take(data,ranks)
    print("--- time after eval ---"  ,(time.time() - time_before_eval))
    # print("image_correspond",image_correspond.shape)

    for i in tqdm(range (image_correspond.shape[0])):

        save_img_2(img_represent_new[i],image_correspond[i],"/media/anlab/DATA/lashinbang-server/fire/FIRe/test_result",ids[i])
    return image_correspond
    # search = np.concatenate((np.array(img_represent).reshape(-1, 1), image_correspond), axis=1)
    # for i in tqdm(range(search.shape[0])):
    #     save_img(search[i],"/media/anlab/data/lashinbang/lashinbang-server/stress_test/fire/FIRe/test_result")

    #########method 1#########################
    # print("--- time after get represent ---"  ,(time.time() - time_before_eval))
    # for i,value in enumerate(tqdm(img_represent)):
    #     # print("value",value)
    #     query=[value]
    #     id=ids[i]
    #     datasets_new=[]
    #     for result in results:
    #         id_check= result[0].split("/")[-1].split("_")[0]
    #         if(id_check==str(id)):
    #             datasets_new=datasets_new+ result[1:]

    #     ranks=eval_asmk(query,datasets_new,net, params['evaluation']['inference'], globals, **params['evaluation']['local_descriptor'])[:,:5]
    #     image_correspond=np.take(datasets_new,ranks)
    #     search = np.concatenate((np.array(query).reshape(-1, 1), image_correspond), axis=1)
        
    #     for i in tqdm(range(search.shape[0])):
    #         save_img(search[i],"/media/anlab/data/lashinbang/lashinbang-server/stress_test/FIRe/test_result")
    # time_after_eval= time.time()
    # print("--- time after eval ---"  ,(time_after_eval - time_before_eval))
    ##########################
def pil_loader(path):
    # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')
def convert_cv2_to_pil(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img=Image.fromarray(img)
    return img
def get_image_represent(list_img_query, list_id, list_result_correspond):
    list_data_full=[]
    ids=[]
    img_represent=[]
    fm_represent_max=[]
    for i,value in enumerate(list_img_query):
        id=list_id[i]
        img= cv2.cvtColor(value, cv2.COLOR_BGR2GRAY)
        fm=variance_of_laplacian(img)
        if(id not in ids):
            ids.append(id)
            fm_represent_max.append(fm)
            img_represent.append(convert_cv2_to_pil(value))
        else:
            if(fm>fm_represent_max[ids.index(id)]):
                fm_represent_max[ids.index(id)]=fm
                img_represent[ids.index(id)]=convert_cv2_to_pil(value)
        list_data_full=list_data_full+list_result_correspond[i]
        # break
    
    list_data_full=list(set(list_data_full))
    return ids,img_represent,list_data_full

def get_rerank_obj(qimages, ids, results,path_foder_save):
    list_id,list_img_query_represent,list_data_full= get_image_represent(qimages, ids, results)
    print("list_data_full",list_data_full[0])
    image_correspond=run(list_img_query_represent,list_data_full,list_id,path_foder_save)
    # print("image_correspond",image_correspond[0])
    result_rerank=[]
    for i,value in enumerate(image_correspond):
        result_query=[]
        for path_data in value:
            dict_result={"image": path_data, "score":0.5}
            result_query.append(dict_result)
        result_rerank.append(result_query)

    return ids,result_rerank
if __name__ == "__main__":
    # path="/media/anlab/data/lashinbang/lashinbang-server/revamptracking/output/5336_new/test/data_full/"
    # with open(f'/media/anlab/data/lashinbang/lashinbang-server/stress_test/revamp_20220504.csv','r') as read:
    #     reader = csv.reader(read)
    #     results=[]
    #     ids=[]
    #     img_represent=[]
    #     fm_represent_max=[]
    #     # fm_save=0
    #     for row in reader:
    #         id=row[2].split("/")[-1].split("_")[0]
    #         fm=caculate_blur(row[2])
    #         if(id not in ids):
    #             ids.append(id)
    #             fm_represent_max.append(fm)
    #             img_represent.append(row[2])
    #         else:
    #             if(fm>fm_represent_max[ids.index(id)]):
    #                 fm_represent_max[ids.index(id)]=fm
    #                 img_represent[ids.index(id)]=row[2]
                
    #         add=[row[2]]
    #         row[3]=row[3].split("|")
    #         row[3]= [path+ value.split("/")[-1] for value in row[3]]
    #         add=add+row[3]
    #         # print(add)
    #         results.append(add)
    # data=os.listdir(path)
    # data=[path+value for value in data]

    # # method 2
    # img_represent_new=[pil_loader(value) for value in img_represent]
    # run(img_represent_new,data,[])

    #method 1
    # run(img_represent,results)

    #### get data
    path="/media/anlab/data/lashinbang/lashinbang-server/revamptracking/output/5336_new/test/data_full/"
    with open(f'/media/anlab/data/lashinbang/lashinbang-server/stress_test/revamp_20220504.csv','r') as read:
        reader = csv.reader(read)
        ids=[]
        qimages=[]
        results=[]
        for row in reader:
            # print("row",row)
            ids.append(row[2].split("/")[-1].split("_")[0])
            qimages.append(cv2.imread(row[2]))
            row[3]=row[3].split("|")
            row[3]= [path+ value.split("/")[-1] for value in row[3]]
            results.append(row[3])
    ######################

    list_id,result_rerank=get_rerank_obj(qimages, ids, results)
 

    
    


