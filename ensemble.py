import os
from sklearn.preprocessing import LabelEncoder
import numpy as np
import nibabel as nib
import pandas as pd
import sklearn
import xgboost as xgb
import matplotlib.pyplot as plt
import warnings
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, ExtraTreesClassifier, ExtraTreesRegressor, GradientBoostingClassifier,GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import roc_auc_score, log_loss
from xgboost import XGBClassifier
from sklearn.utils import shuffle

def crop3d(input_img, boundaries=[38, 29, 0, 202, 224, 149]):
        output_img = input_img[38:203,29:225,0:150]
        return(output_img)

def test():
    #To get the names of the subject's data
    from medpy import metric
    print("here")
    root = '/media/hdd2/pkao/brats2018/output/valid'
    root_dir = '/media/hdd1/pkao/brats2018/validation'
    file_list = os.path.join(root_dir, 'test.txt')
    names = open(file_list).read().splitlines()
    submission_name = 'test'
    ans_avg = 0
    ans_xgb = 0
    submission_dir = os.path.join('submissions_class4_avg_level3', submission_name+'_uint8', 'valid')
    if not os.path.exists(submission_dir):
        os.makedirs(submission_dir)
    for name in names[:]:
        x_train = []
        y_train = []
        print(name)
        x_npy = []
        oname = os.path.join(submission_dir, name+'.nii.gz')
        if 'HGG' in name or 'LGG' in name:
            name = name[4:]
            print(name)
        for k, model in enumerate(models):
            fname = os.path.join(root, models[k], name+'_preds.npy')
            prob_map_float32 = np.load(fname)
            x_npy_channel = []
            for i in range(5):
                x_npy_channel.append(crop3d(prob_map_float32[i]))
            x_npy.append(x_npy_channel)

        x_train.append(x_npy)
        x_tr = np.reshape(x_train, (10,165,196,150))
        x = np.reshape(np.transpose(x_tr), (4851000,10))
        xg_test = xgb.DMatrix(x)
        pre=model_2_v2.predict(xg_test)
        
        preds = []
        for i in pre:
            if (i==1):
                preds.append(4)
            else:
                preds.append(0)
        outimg = np.reshape(preds,(150,196,165))
        rimg = np.transpose(outimg).astype('uint8')
        output = np.zeros((240,240,155))       
        for I in range(38,203):
            for J in range(29,225):
                for K in range(0,150):
                    output[I][J][K] = rimg[I-38][J-29][K]
               
        ri_img = nib.Nifti1Image(output.astype('uint8'), None)
        nib.save(ri_img, oname)

os.environ['CUDA_VISIBLE_DEVICES'] = '1'

settings = {
        'five_deepmedics_five_unets':{
        'models': ['averageProbMaps/average_three_deepmedics', 'averageProbMaps/average_three_unets'
]
        }
    }

#set files path
root = '/media/hdd2/pkao/brats2018/output/training'
models = settings['five_deepmedics_five_unets']['models']
root_dir = '/media/hdd1/pkao/brats2018/training'
file_list = os.path.join(root_dir, 'all.txt')
names = open(file_list).read().splitlines()           
# xgbt = xgb(max_depth=3,learning_rate=0.1, n_estimators=100, objective="rank:pairwise", n_jobs=1, booster="gbtree", random_state=1, colsample_bytree=1, scale_pos_weight=49)

params = {'max_depth':3,'learning_rate':0.9, 'n_estimators':300, 'objective':"binary:hinge", 'n_jobs':1, 'booster':"gbtree", 'random_state':1, 'colsample_bytree':1}
x_d = []
y_d = []
for name in names[:185]:
    x_train = []
    y_train = []
    x_npy = []
    if 'HGG' in name or 'LGG' in name:
        name = name[4:]
    for k, model in enumerate(models):
        fname = os.path.join(root, models[k], name+'_preds.npy')
        prob_map_float32 = np.load(fname)
        x_npy_channel = []
        for i in range(5):
            x_npy_channel.append(crop3d(prob_map_float32[i]))
        x_npy.append(x_npy_channel)
    y_ensemble = np.zeros((5,165,196,150))
    for i in range(2):
        y_ensemble = np.add(y_ensemble, x_npy[i])
    y_en = y_ensemble
    y_e = y_en.argmax(0).astype('uint8')
    y_fname = os.path.join('/home/shailja/Y_labels', name + '_seg.nii.gz')
    img = nib.load(y_fname)
    y_train.append(crop3d(img.get_data()))
    y_t = np.array(np.transpose(y_train)).flatten()
    y =[]
    e = np.transpose(y_e).flatten()
    for i in y_t:
        if (i == 4):
            y.append(1)
        else:
            y.append(0)
    print(sum(y))            
    x_train.append(x_npy)

    x_tr = np.reshape(x_train, (10,165,196,150))
    x = np.reshape(np.transpose(x_tr), (4851000,10))
    
    c=0
    for i in range(len(x)):
        if (e[i] != 0 and e[i] != 2):
            x_d.append(x[i])
            y_d.append(y[i])
        

print(len(x_d), len(y_d))
model_2_v2 = xgb.train(params, xgb.DMatrix(np.array(x_d), label=np.array(y_d)), 30)
model_2_v2.save_model('model_3_1.model')

print("training done.")
test()
