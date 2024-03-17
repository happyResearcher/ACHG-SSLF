import pickle
import pandas as pd

#%% create metadata of the original dataset
# Nodes
# User: userlD (36063), titleDetail(56), company(49), jobcode(13), jobFamily(70), rollup1(12), rollup2(111), rollup3(12)
df = pd.read_csv('./data/UserNodes.csv')
User_features=df.rename(columns={'personID':'userID', 'personBusinessTitleDetail': 'titleDetail', 'personCompany':'company',
'personJobCode':'jobcode', 'personJobFamily':'jobFamily', 'personRollup1':'rollup1', 'personRollup2':'rollup2',
                              'personRollup3':'rollup3'})


User_features = User_features.reindex(columns=['userID', 'titleDetail', 'company', 'jobcode', 'jobFamily',
       'rollup1', 'rollup2', 'rollup3']).drop_duplicates().reset_index(drop=True)

User_features['userID'].nunique() #36063

# Resource: resourceID(33,252), resourceType(3: group, system, host)
Resource_features = pd.read_csv('./data/ResourceNodes.csv').drop_duplicates().reset_index(drop=True)
Resource_features['resourceID'].nunique() # 33252

# Title: titleID(4979)
df = pd.read_csv('./data/TitleNodes.csv')
Title_feature = df[['personBusinessTitle']].drop_duplicates()
Title_feature=Title_feature.rename(columns={'personBusinessTitle':'titleID'}).reset_index(drop=True)
Title_feature['titleID'].nunique() # 4979

# Manager: managerID (3,207)
Manager_feature = pd.read_csv('./data/ManagerNodes.csv').drop_duplicates().reset_index(drop=True)
Manager_feature['managerID'].nunique() #3207

# Department: deptID (405)
Department_feature = pd.read_csv('./data/DepartmentNodes.csv').drop_duplicates().rename(columns={'personDeptID':'deptID'
                                                                                                 }).reset_index(drop=True)
Department_feature['deptID'].nunique() #405

# Relationships
# User -> HAS_P_ACCESS -> Resource #595506
df = pd.read_csv('./data/User-HAVE_POTENTIAL_ACCESS-Resource.csv')
HAS_P_ACCESS = df.iloc[:,0:2].rename(columns={'personID':'HAS_P_ACCESS_HeadID',
                                              'resourceID':'HAS_P_ACCESS_TailID'}).drop_duplicates().reset_index(drop=True)

#User -> HAS_TITLE -> Title #36063
df = pd.read_csv('./data/User-WORK_AS-Title.csv')
HAS_TITLE = df.iloc[:,0:2].rename(columns={'personID':'HAS_TITLE_HeadID',
                                            'personBusinessTitle':'HAS_TITLE_TailID'}).drop_duplicates().reset_index(drop=True)

# User -> HAS_MANAGER -> Manager #36063
df = pd.read_csv('./data/User-SUPERVISED_BY-Manager.csv')
HAS_MANAGER = df.iloc[:,0:2].rename(columns={'personID':'HAS_MANAGER_HeadID',
                                            'managerID':'HAS_MANAGER_TailID'}).drop_duplicates().reset_index(drop=True)

#User -> HAS_DEPT-> Department #36063
df = pd.read_csv('./data/User-BELONG_TO-Department.csv')
HAS_DEPT = df.iloc[:,0:2].rename(columns={'personID':'HAS_DEPT_HeadID',
                                            'personDeptID':'HAS_DEPT_TailID'}).drop_duplicates().reset_index(drop=True)

acess_control_metadata_dict = {
    'User_features': User_features,
    'Resource_features': Resource_features,
    'Title_feature': Title_feature,
    'Manager_feature': Manager_feature,
    'Department_feature': Department_feature,
    'HAS_P_ACCESS': HAS_P_ACCESS,
    'HAS_TITLE': HAS_TITLE,
    'HAS_MANAGER': HAS_MANAGER,
    'HAS_DEPT': HAS_DEPT
}


with open('./data/acess_control_metadata_dict.pickle', 'wb') as f:
    pickle.dump(acess_control_metadata_dict, f)


### Mapping node indices to contiguous integers before creating HeteroData
#%% User indices mapping and replace
User_id_mapping = {}
for i in range(len(User_features)):
    User_id_mapping[User_features.userID[i]]= i

# replace the id in related relationships
User_features['mappedID'] = User_features['userID'].replace(User_id_mapping) # # Create a new column with mapped IDs
HAS_P_ACCESS['HAS_P_ACCESS_HeadID'].replace(User_id_mapping, inplace=True) # userID replace
HAS_TITLE['HAS_TITLE_HeadID'].replace(User_id_mapping, inplace=True) # userID replace
HAS_MANAGER['HAS_MANAGER_HeadID'].replace(User_id_mapping, inplace=True) # userID replace
HAS_DEPT['HAS_DEPT_HeadID'].replace(User_id_mapping, inplace=True) # userID replace


#%% Resource indices mapping and replace
Resource_id_mapping = {}
for i in range(len(Resource_features)):
    Resource_id_mapping[Resource_features.resourceID[i]]= i

# replace the id in related relationships
Resource_features['mappedID'] = Resource_features['resourceID'].replace(Resource_id_mapping) # # Create a new column with mapped IDs
HAS_P_ACCESS['HAS_P_ACCESS_TailID'].replace(Resource_id_mapping, inplace=True) # resourceID replace

#%% Title indices mapping and replace
Title_id_mapping = {}
for i in range(len(Title_feature)):
    Title_id_mapping[Title_feature.titleID[i]]= i

# replace the id in related relationships
Title_feature['mappedID'] = Title_feature['titleID'].replace(Title_id_mapping) # # Create a new column with mapped IDs
HAS_TITLE['HAS_TITLE_TailID'].replace(Title_id_mapping, inplace=True) # titleID replace

#%% Manager indices mapping and replace
Manager_id_mapping = {}
for i in range(len(Manager_feature)):
    Manager_id_mapping[Manager_feature.managerID[i]]= i

# replace the id in related relationships
Manager_feature['mappedID'] = Manager_feature['managerID'].replace(Manager_id_mapping) # # Create a new column with mapped IDs
HAS_MANAGER['HAS_MANAGER_TailID'].replace(Manager_id_mapping, inplace=True) # managerID replace

#%% Department indices mapping and replace
Department_id_mapping = {}
for i in range(len(Department_feature)):
    Department_id_mapping[Department_feature.deptID[i]]= i

# replace the id in related relationships
Department_feature['mappedID'] = Department_feature['deptID'].replace(Department_id_mapping) # # Create a new column with mapped IDs
HAS_DEPT['HAS_DEPT_TailID'].replace(Department_id_mapping, inplace=True) # deptID replace

#%%
acess_control_metadata_dict_remapping = {
    'User_features': User_features,
    'Resource_features': Resource_features,
    'Title_feature': Title_feature,
    'Manager_feature': Manager_feature,
    'Department_feature': Department_feature,
    'HAS_P_ACCESS': HAS_P_ACCESS,
    'HAS_TITLE': HAS_TITLE,
    'HAS_MANAGER': HAS_MANAGER,
    'HAS_DEPT': HAS_DEPT
}

with open('./data/acess_control_metadata_dict_remapping.pickle', 'wb') as f:
    pickle.dump(acess_control_metadata_dict, f)

