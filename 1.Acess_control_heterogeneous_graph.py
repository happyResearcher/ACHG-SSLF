import pickle
import pandas as pd
import torch
import category_encoders as ce
from torch_geometric.data import HeteroData
from torch_geometric.data import download_url, extract_zip
import pandas as pd
import numpy as np
import torch
from torch import Tensor
print(torch.__version__)
from early_stop_v1 import EarlyStopping

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt

from torch_geometric.data import HeteroData
import torch_geometric.transforms as T
import tqdm
import torch.nn.functional as F

# Install required packages.
import os
os.environ['TORCH'] = torch.__version__

#%% load metadata, nodes and relationships
## load metadata
with open('./data/acess_control_metadata_dict_remapping.pickle', 'rb') as f:
    acess_control_metadata_dict = pickle.load(f)

## load nodes and relationships
User_features=acess_control_metadata_dict["User_features"]
Resource_features=acess_control_metadata_dict["Resource_features"]
Title_feature=acess_control_metadata_dict["Title_feature"]
Manager_feature=acess_control_metadata_dict["Manager_feature"]
Department_feature=acess_control_metadata_dict["Department_feature"]
HAS_P_ACCESS=acess_control_metadata_dict["HAS_P_ACCESS"]
HAS_TITLE=acess_control_metadata_dict["HAS_TITLE"]
HAS_MANAGER=acess_control_metadata_dict["HAS_MANAGER"]
HAS_DEPT=acess_control_metadata_dict["HAS_DEPT"]

#%% encoding features of User
#user features BinaryEncoder # 36063*38
user_feature_columns = ['titleDetail', 'company', 'jobcode', 'jobFamily',
       'rollup1', 'rollup2', 'rollup3']
encoder = ce.BinaryEncoder(cols=user_feature_columns)
User_features = encoder.fit_transform(User_features)

# one-hot encode resource features
# Resource_features['resourceType'] = Resource_features['resourceType'].astype('category').cat.codes # category encoding
one_hot_encoded  = pd.get_dummies(Resource_features['resourceType'], prefix='resourceType')
Resource_features = pd.concat([Resource_features, one_hot_encoded], axis=1).drop('resourceType', axis=1)

#%% Create a HeteroData object
data = HeteroData()
# nodes feature population
user_feat = torch.tensor(User_features.iloc[:,1:-1].values, dtype=torch.float32) # (36063, 38)
data['User'].x = user_feat
data['User'].node_id = torch.tensor(User_features['mappedID'].values)

data['Resource'].x =torch.tensor(Resource_features.iloc[:,2:].values, dtype=torch.float32) # (33252, 3)
data['Resource'].node_id = torch.tensor(Resource_features['mappedID'].values)

data['Title'].node_id =torch.tensor(Title_feature['mappedID'].values) #torch.Size([4979])
data['Manager'].node_id =torch.tensor(Manager_feature['mappedID'].values) #torch.Size([3207])
data['Department'].node_id =torch.tensor(Department_feature['mappedID'].values) #torch.Size([405])

# relationship population
data['User', 'HAS_P_ACCESS', 'Resource'].edge_index = torch.transpose(
    torch.tensor(HAS_P_ACCESS[['HAS_P_ACCESS_HeadID','HAS_P_ACCESS_TailID']].values),0,1) # [2, 595506]
data['User', 'HAS_TITLE', 'Title'].edge_index = torch.transpose(
    torch.tensor(HAS_TITLE[['HAS_TITLE_HeadID','HAS_TITLE_TailID']].values),0,1) # [2, 36063]
data['User', 'HAS_MANAGER', 'Manager'].edge_index = torch.transpose(
    torch.tensor(HAS_MANAGER[['HAS_MANAGER_HeadID','HAS_MANAGER_TailID']].values),0,1) # [2, 36063]
data['User', 'HAS_DEPT', 'Department'].edge_index = torch.transpose(
    torch.tensor(HAS_DEPT[['HAS_DEPT_HeadID','HAS_DEPT_TailID']].values),0,1) # [2, 36063]

# Save the heterogeneous graph data
torch.save(data, './data/acess_control_heterogeneous_graph.pt')
## Load the heterogeneous graph data
# data = torch.load('./data/acess_control_heterogeneous_graph.pt')
