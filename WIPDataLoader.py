import numpy as np
import warnings
import os
from torch.utils.data import Dataset
from sklearn.preprocessing import normalize
from torch.utils.data.dataloader import DataLoader
from torch.utils.data.sampler import SubsetRandomSampler

# import matplotlib.pyplot as plt
# import seaborn as sns
warnings.filterwarnings('ignore')


label_to_idx = {"standing" : 0,"walking": 1, "jogging": 2, "jumping": 3, "squat down": 4,
                "squating": 5, "squatup": 6, "forward": 7,
                "backoff": 7}

def pc_normalize(pc):
    centroid = np.mean(pc, axis=0)  
    pc = pc - centroid  
    m = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    pc = pc / m 
    return pc

def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[12]) # 12 is feature size
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[12])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def load_locomotion_loso_data(DATA_PATH, out_user_name):
    """
    leave one subject out method
    leave 'out_user' as target domain, and all other users as source domain
    """
    
    Xlist = []
    Ylist = []
    username_list = ['S1', 'S2', 'S3', 'S4', 'S5', 'S6', 'S7', 'S8', 'S9','S10', 'S11', 'S12', 'S13']
    out_user_index = username_list.index(out_user_name)
    
    for user in username_list:
        for c in os.listdir(DATA_PATH):
            if(user in c):
                if(c.split('_')[1][0] == 'X'):
                    Xlist.append(np.load(os.path.join(DATA_PATH,c)))
                elif(c.split('_')[1][0] == 'Y'):
                    Ylist.append(np.load(os.path.join(DATA_PATH,c)))
    Xlist = np.array(Xlist,dtype=object)
    Ylist = np.array(Ylist,dtype=object) 
     
    
    mask = np.ones((Xlist.shape[0],), dtype=np.bool)
    mask[out_user_index] = False
    
    Xtrain = np.concatenate(Xlist[mask], axis=0) 
    Ytrain = np.concatenate(Ylist[mask], axis=0) 
    Xtest = Xlist[~mask][0] 
    Ytest = Ylist[~mask][0] 

    max_feature = np.max(Xtrain, axis=0)
    min_feature = np.min(Xtrain, axis=0)
    
    Xtrain = (Xtrain - min_feature) / (max_feature - min_feature)
    Xtest = (Xtest - min_feature) / (max_feature - min_feature)
    
    # standing-walking mask
    standing_walking_mask = False
    if standing_walking_mask:
        sw_mask = np.zeros((Ytrain.shape[0],), dtype=np.bool)
        sw_mask[Ytrain==0] = True
        sw_mask[Ytrain==1] = True
        Xtrain = Xtrain[sw_mask]
        Ytrain = Ytrain[sw_mask]
        
        sw_mask = np.zeros((Ytest.shape[0],), dtype=np.bool)
        sw_mask[Ytest==0] = True
        sw_mask[Ytest==1] = True
        Xtest = Xtest[sw_mask]
        Ytest = Ytest[sw_mask]

    print('xytrain xytest shape',Xtrain.shape, Ytrain.shape, Xtest.shape, Ytest.shape)
    print('unique labels ',np.unique(Ytrain),np.unique(Ytest))
    
    SOURCE_DATASET = LocomotionLOSODataLoader(Xtrain,Ytrain,'source')
    TARGET_DATASET = LocomotionLOSODataLoader(Xtest,Ytest,'target')
    return SOURCE_DATASET, TARGET_DATASET


class LocomotionLOSODataLoader(Dataset):
    def __init__(self, X, y,partition='source'):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.int)
        self.partition = partition
        
        self.num_examples = self.X.shape[0]
        self.train_ind = np.asarray([i for i in range(self.num_examples) if i % 10 < 8]).astype(np.int)
            
        self.val_ind = np.asarray([i for i in range(self.num_examples) if i % 10 >= 8]).astype(np.int)
        np.random.shuffle(self.val_ind)

    def __len__(self) -> int:
        return self.num_examples
    

    def __getitem__(self, index: int):
        if self.partition == 'source':
            x = self.X[index]
            x = translate_pointcloud(x)
            return x , self.y[index]
        else:
            return self.X[index], self.y[index]


if __name__ == '__main__':
    import torch
    dataroot = 'Locomotion/data/16frame/step3'
    target_username = 'taoshu'
    batch_size = 64
    
    src_trainset ,trgt_trainset = load_locomotion_loso_data(dataroot,target_username)
    
    

    
    