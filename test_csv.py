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
from evaluate import run
path="/media/anlab/data/lashinbang/lashinbang-server/revamptracking/output/5336_new/bad/data/"

def variance_of_laplacian(image):
	# compute the Laplacian of the image and then return the focus
	# measure, which is simply the variance of the Laplacian
	return cv2.Laplacian(image, cv2.CV_64F).var()

def caculate_blur(imagePath):
    image = cv2.imread(imagePath)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fm = variance_of_laplacian(gray)
    return fm

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







with open(f'/media/anlab/data/lashinbang/lashinbang-server/stress_test/revamp_20220504.csv','r') as read:
    reader = csv.reader(read)
    results=[]
    ids=[]
    img_represent=[]
    fm_represent_max=[]
    # fm_save=0
    for row in reader:
        id=row[2].split("/")[-1].split("_")[0]
        fm=caculate_blur(row[2])
        if(id not in ids):
            ids.append(id)
            fm_represent_max.append(fm)
            img_represent.append(row[2])
        else:
            if(fm>fm_represent_max[ids.index(id)]):
                fm_represent_max[ids.index(id)]=fm
                img_represent[ids.index(id)]=row[2]
            
        add=[row[2]]
        row[3]=row[3].split("|")
        row[3]= [path+ value.split("/")[-1] for value in row[3]]
        add=add+row[3]
        # print(add)
        results.append(add)


    # print(results[0])
    # print(ids)
    # print("img_represent",np.array(img_represent))
    # exit()
    # print("result",result)

    # for value in tqdm(result):
    #     save_img(value,"/media/anlab/data/lashinbang/lashinbang-server/revamptracking/output/5336_new/bad/search")
    # id=75
    # query=[f"/media/anlab/data/lashinbang/lashinbang-server/revamptracking/output/5336_new/bad/{id}_279.jpg"]
    for i,value in enumerate(img_represent):
        print("value",value)
        query=[value]
        id=ids[i]
        datasets_new=[]
        for result in results:
            id_check= result[0].split("/")[-1].split("_")[0]
            if(id_check==str(id)):
                datasets_new=datasets_new+ result[1:]
        
        
        # print("datasets_new",datasets_new)
        # print("query",query)
        # loader=default_loader
        # img_query = loader(query[0])

        with open("/media/anlab/data/lashinbang/lashinbang-server/stress_test/FIRe/query.pickle","wb") as f:
            pickle.dump(query,f)

        with open("/media/anlab/data/lashinbang/lashinbang-server/stress_test/FIRe/dataset.pickle","wb") as f:
            pickle.dump(datasets_new,f)

        # exit()

        run()
        with open("/media/anlab/data/lashinbang/lashinbang-server/stress_test/FIRe/ranks.pickle","rb") as f:
            ranks=pickle.load(f)[:,:5]
        print("ranks",ranks)
        print("dataset",len(datasets_new),ranks.shape)

        image_correspond=np.take(datasets_new,ranks)
        search = np.concatenate((np.array(query).reshape(-1, 1), image_correspond), axis=1)
        
        for i in tqdm(range(search.shape[0])):
            save_img(search[i],"/media/anlab/data/lashinbang/lashinbang-server/stress_test/FIRe/test_result")
    # break



    
    

    
    