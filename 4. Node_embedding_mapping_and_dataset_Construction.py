import numpy as np
import torch
print(torch.__version__)
import category_encoders as ce

import os
os.environ['TORCH'] = torch.__version__
import pandas as pd
import pickle

#%% load the node embeddings
#% ablation setting
ablation_settings = ['None'] #'All','Title','Department','Manager','None'
batch_size= 1024
hidden_channels=[16,32,128,256] #16,32,64,128,256
Epoch=1000
imbalance_settings = ['55','37','19','original'] # '55','37','19','original'

with open('./data/acess_control_metadata_dict.pickle', 'rb') as f:
    acess_control_metadata_dict = pickle.load(f)

for hidden_channel in hidden_channels:

    for ablation_setting in ablation_settings:
        setting = ablation_setting + '_BS_'+ str(batch_size) + '_HC_' + str(hidden_channel)

        embeddings_dict = torch.load('./data/embedding/' + setting + '_Node_embeddings.pt')
        User_embedding = embeddings_dict['User_embedding'].detach().numpy()
        Resource_embedding = embeddings_dict['Resource_embedding'].detach().numpy()

        for imbalance_setting in imbalance_settings:
            if imbalance_setting in ['55','37','19']:
                df = pd.read_pickle('data/' + 'access_log_'+ imbalance_setting +'_with_linkPredictionFeatures.pkl')
            elif imbalance_setting == 'original':
                df = pd.read_pickle('data/' + 'access_log_original.pkl')
            data = df[['ACTION', 'PERSON_ID','TARGET_NAME']]

            # Filter rows where 'TARGET_NAME' is less than or equal to 33252
            data = data[data['TARGET_NAME'] < len(Resource_embedding)]

            User_features = acess_control_metadata_dict["User_features"]
            Resource_features = acess_control_metadata_dict["Resource_features"]

            # user features BinaryEncoder # 36063*38
            user_feature_columns = ['titleDetail', 'company', 'jobcode', 'jobFamily',
                                    'rollup1', 'rollup2', 'rollup3']
            encoder = ce.BinaryEncoder(cols=user_feature_columns)
            User_features = encoder.fit_transform(User_features)
            User_feat = User_features.iloc[:, 1:].to_numpy()

            # Resource_features['resourceType'] = Resource_features['resourceType'].astype('category').cat.codes # category encoding
            one_hot_encoded = pd.get_dummies(Resource_features['resourceType'], prefix='resourceType')
            Resource_features = pd.concat([Resource_features, one_hot_encoded], axis=1).drop('resourceType', axis=1)
            Resource_feat = Resource_features.iloc[:, 1:].to_numpy().astype(int)


            y=data['ACTION'].to_numpy()

            User_id_mapping = {}
            for i in range(len(User_features)):
                User_id_mapping[User_features.userID[i]] = i

            # Extract the User IDs and convert them to a NumPy array and then get the x_User_embeddings
            user_ids =  data['PERSON_ID'].replace(User_id_mapping).to_numpy()   #

            user_embeddings_list = [User_embedding[idx, :] for idx in user_ids]
            x_User_embeddings = np.vstack(user_embeddings_list)

            User_feat_list = [User_feat[idx, :] for idx in user_ids]
            x_User_feat = np.vstack(User_feat_list)


            Resource_id_mapping = {}
            for i in range(len(Resource_features)):
                Resource_id_mapping[Resource_features.resourceID[i]] = i

            # Extract the User IDs and convert them to a NumPy array and then get the x_User_embeddings
            resource_ids =  data['TARGET_NAME'].replace(Resource_id_mapping).to_numpy()   #

            resource_embeddings_list = [Resource_embedding[idx, :] for idx in resource_ids]
            x_Resource_embeddings = np.vstack(resource_embeddings_list)

            Resource_feat_list = [Resource_feat[idx, :] for idx in resource_ids]
            x_Resource_feat = np.vstack(Resource_feat_list)




            dataset_dict = {
                'y': y,
                'x_User_embeddings':x_User_embeddings,
                'x_User_feat':x_User_feat,
                'x_Resource_embeddings':x_Resource_embeddings,
                'x_Resource_feat': x_Resource_feat,
            }

            with open('./data/dataset/'+ ablation_setting +'_'+ imbalance_setting + '_HC_' + str(hidden_channel)+'_dataset_dict.pickle', 'wb') as f:
                pickle.dump(dataset_dict, f)

