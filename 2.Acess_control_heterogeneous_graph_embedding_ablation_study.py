import pickle
import pandas as pd
import torch
import category_encoders as ce
from torch_geometric.data import HeteroData
from torch_geometric.data import download_url, extract_zip
import pandas as pd
import csv
import numpy as np
import torch
from torch import Tensor
print(torch.__version__)
from early_stop_v1 import EarlyStopping
import time

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

#%% ## Creating a Heterogeneous Link-level GNN
from torch_geometric.nn import SAGEConv, to_hetero
import torch.nn.functional as F

class GNN(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()

        self.conv1 = SAGEConv(hidden_channels, hidden_channels)
        self.conv2 = SAGEConv(hidden_channels, hidden_channels)

    def forward(self, x: Tensor, edge_index: Tensor) -> Tensor:
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return x

# Our final classifier applies the dot-product between source and destination
# node embeddings to derive edge-level predictions:
class Classifier(torch.nn.Module):
    def forward(self, x_User: Tensor, x_Resource: Tensor, edge_label_index: Tensor) -> Tensor:
        # Convert node embeddings to edge-level representations:
        edge_feat_User = x_User[edge_label_index[0]]
        edge_feat_Resource = x_Resource[edge_label_index[1]]

        # Apply dot-product to get a prediction per supervision edge:
        return (edge_feat_User * edge_feat_Resource).sum(dim=-1)


class Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn
        # embedding matrices for different types of nodes:
        self.User_lin = torch.nn.Linear(38, hidden_channels)
        self.User_emb = torch.nn.Embedding(data["User"].num_nodes, hidden_channels)
        self.Resource_lin = torch.nn.Linear(3, hidden_channels)
        self.Resource_emb = torch.nn.Embedding(data["Resource"].num_nodes, hidden_channels)
        self.Title_emb = torch.nn.Embedding(data["Title"].num_nodes, hidden_channels)
        self.Manager_emb = torch.nn.Embedding(data["Manager"].num_nodes, hidden_channels)
        self.Department_emb = torch.nn.Embedding(data["Department"].num_nodes, hidden_channels)


        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)

        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())

        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
          "User": self.User_lin(data["User"].x) + self.User_emb(data["User"].node_id),
          "Resource": self.Resource_lin(data["Resource"].x) + self.Resource_emb(data["Resource"].node_id),
          "Title": self.Title_emb(data["Title"].node_id),
          "Manager": self.Manager_emb(data["Manager"].node_id),
          "Department": self.Department_emb(data["Department"].node_id),
        }

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["User"],
            x_dict["Resource"],
            data['User', 'HAS_P_ACCESS', 'Resource'].edge_label_index,
        )

        return pred,x_dict

class Title_Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn
        # embedding matrices for different types of nodes:
        self.User_lin = torch.nn.Linear(38, hidden_channels)
        self.User_emb = torch.nn.Embedding(data["User"].num_nodes, hidden_channels)
        self.Resource_lin = torch.nn.Linear(3, hidden_channels)
        self.Resource_emb = torch.nn.Embedding(data["Resource"].num_nodes, hidden_channels)
        self.Title_emb = torch.nn.Embedding(data["Title"].num_nodes, hidden_channels)

        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)

        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())

        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
          "User": self.User_lin(data["User"].x) + self.User_emb(data["User"].node_id),
          "Resource": self.Resource_lin(data["Resource"].x) + self.Resource_emb(data["Resource"].node_id),
          "Title": self.Title_emb(data["Title"].node_id),
        }

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["User"],
            x_dict["Resource"],
            data['User', 'HAS_P_ACCESS', 'Resource'].edge_label_index,
        )
        return pred,x_dict

class Manager_Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn
        # embedding matrices for different types of nodes:
        self.User_lin = torch.nn.Linear(38, hidden_channels)
        self.User_emb = torch.nn.Embedding(data["User"].num_nodes, hidden_channels)
        self.Resource_lin = torch.nn.Linear(3, hidden_channels)
        self.Resource_emb = torch.nn.Embedding(data["Resource"].num_nodes, hidden_channels)
        self.Manager_emb = torch.nn.Embedding(data["Manager"].num_nodes, hidden_channels)


        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)

        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())

        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
          "User": self.User_lin(data["User"].x) + self.User_emb(data["User"].node_id),
          "Resource": self.Resource_lin(data["Resource"].x) + self.Resource_emb(data["Resource"].node_id),
          "Manager": self.Manager_emb(data["Manager"].node_id),
        }

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["User"],
            x_dict["Resource"],
            data['User', 'HAS_P_ACCESS', 'Resource'].edge_label_index,
        )
        return pred,x_dict

class Department_Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn
        # embedding matrices for different types of nodes:
        self.User_lin = torch.nn.Linear(38, hidden_channels)
        self.User_emb = torch.nn.Embedding(data["User"].num_nodes, hidden_channels)
        self.Resource_lin = torch.nn.Linear(3, hidden_channels)
        self.Resource_emb = torch.nn.Embedding(data["Resource"].num_nodes, hidden_channels)
        self.Department_emb = torch.nn.Embedding(data["Department"].num_nodes, hidden_channels)


        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)

        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())

        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
          "User": self.User_lin(data["User"].x) + self.User_emb(data["User"].node_id),
          "Resource": self.Resource_lin(data["Resource"].x) + self.Resource_emb(data["Resource"].node_id),
          "Department": self.Department_emb(data["Department"].node_id),
        }

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["User"],
            x_dict["Resource"],
            data['User', 'HAS_P_ACCESS', 'Resource'].edge_label_index,
        )
        return pred,x_dict

class None_Model(torch.nn.Module):
    def __init__(self, hidden_channels):
        super().__init__()
        # Since the dataset does not come with rich features, we also learn
        # embedding matrices for different types of nodes:
        self.User_lin = torch.nn.Linear(38, hidden_channels)
        self.User_emb = torch.nn.Embedding(data["User"].num_nodes, hidden_channels)
        self.Resource_lin = torch.nn.Linear(3, hidden_channels)
        self.Resource_emb = torch.nn.Embedding(data["Resource"].num_nodes, hidden_channels)


        # Instantiate homogeneous GNN:
        self.gnn = GNN(hidden_channels)

        # Convert GNN model into a heterogeneous variant:
        self.gnn = to_hetero(self.gnn, metadata=data.metadata())

        self.classifier = Classifier()

    def forward(self, data: HeteroData) -> Tensor:
        x_dict = {
          "User": self.User_lin(data["User"].x) + self.User_emb(data["User"].node_id),
          "Resource": self.Resource_lin(data["Resource"].x) + self.Resource_emb(data["Resource"].node_id),
        }

        # `x_dict` holds feature matrices of all node types
        # `edge_index_dict` holds all edge indices of all edge types
        x_dict = self.gnn(x_dict, data.edge_index_dict)
        pred = self.classifier(
            x_dict["User"],
            x_dict["Resource"],
            data['User', 'HAS_P_ACCESS', 'Resource'].edge_label_index,
        )
        return pred,x_dict


#%% ## Training a Heterogeneous Link-level GNN
def train(train_loader,val_loader):
    total_loss_train = total_examples_train = 0
    total_loss_val = total_examples_val = 0
    for sampled_data_train, sampled_data_val in zip(tqdm.tqdm(train_loader),tqdm.tqdm(val_loader)):
        optimizer.zero_grad()
        sampled_data_train.to(device)
        sampled_data_val.to(device)

        pred_train,_ = model(sampled_data_train)
        pred_val,_ = model(sampled_data_val)

        ground_truth_train = sampled_data_train['User', 'HAS_P_ACCESS', 'Resource'].edge_label
        ground_truth_val = sampled_data_val['User', 'HAS_P_ACCESS', 'Resource'].edge_label

        loss_train = F.binary_cross_entropy_with_logits(pred_train, ground_truth_train)
        loss_val = F.binary_cross_entropy_with_logits(pred_val, ground_truth_val)

        loss_train.backward()
        optimizer.step()

        total_loss_train += float(loss_train) * pred_train.numel()
        total_examples_train += pred_train.numel()

        total_loss_val += float(loss_val) * pred_val.numel()
        total_examples_val += pred_val.numel()

    epoch_loss_train = total_loss_train / total_examples_train
    epoch_loss_val = total_loss_val / total_examples_val

    return epoch_loss_train,epoch_loss_val


#%% ablation setting
ablation_settings = ['None'] #'All','Title','Department','Manager','None'
batch_size= 1024
hidden_channels=[256] #16,32,64,128
Epoch=1000

for hidden_channel in hidden_channels:

    for ablation_setting in ablation_settings:

        setting = ablation_setting + '_BS_'+ str(batch_size) + '_HC_' + str(hidden_channel)

    ## Load the heterogeneous graph data
        data = torch.load('./data/acess_control_heterogeneous_graph.pt')
        print('Ablation_setting: ', ablation_setting)
        if ablation_setting == 'Title':
            del data['Manager']
            del data['Department']
            del data["User", "HAS_MANAGER", "Manager"]
            del data["User", "HAS_DEPT", "Department"]
        elif ablation_setting == 'Department':
            del data['Title']
            del data['Manager']
            del data["User", "HAS_TITLE", "Title"]
            del data["User", "HAS_MANAGER", "Manager"]
        elif ablation_setting == 'Manager':
            del data['Title']
            del data['Department']
            del data["User", "HAS_TITLE", "Title"]
            del data["User", "HAS_DEPT", "Department"]
        elif ablation_setting == 'None':
            del data['Title']
            del data['Department']
            del data['Manager']
            del data["User", "HAS_TITLE", "Title"]
            del data["User", "HAS_DEPT", "Department"]
            del data["User", "HAS_MANAGER", "Manager"]
        data = T.ToUndirected()(data)
        print(data)


        #%% Defining Edge-level Training Splits
        transform = T.RandomLinkSplit(
            num_val=0.1,
            num_test=0,
            disjoint_train_ratio=0.3, # Across the training edges, we use 70% of edges for message passing, and 30% of edges for supervision.
            neg_sampling_ratio=2.0,
            add_negative_train_samaples=False, ## No negative edges added: Whether to add negative
                    # training samples for link prediction.
                    # If the model already performs negative sampling, then the option
                    # should be set to :obj:`False`.
                    # Otherwise, the added negative samples will be the same across
                    # training iterations unless negative sampling is performed again.
                    # (default: :obj:`True`)
            edge_types=('User', 'HAS_P_ACCESS', 'Resource'),
            rev_edge_types=('Resource', 'rev_HAS_P_ACCESS', 'User'),
        )

        train_data, val_data, test_data = transform(data)
        print("Training data:")
        print("==============")
        print(train_data)
        print()
        print("Validation data:")
        print("================")
        print(val_data)


        assert train_data['User', 'HAS_P_ACCESS', 'Resource'].num_edges == 375170  #375170 = 595506 * 0.9 * 0.7
        assert train_data['User', 'HAS_P_ACCESS', 'Resource'].edge_label_index.size(1) == 160786 # 595506 * 0.9 * 0.3
        assert train_data['Resource', 'rev_HAS_P_ACCESS', 'User'].num_edges == 375170
        # No negative edges added:
        assert train_data['User', 'HAS_P_ACCESS', 'Resource'].edge_label.min() == 1
        assert train_data['User', 'HAS_P_ACCESS', 'Resource'].edge_label.max() == 1

        assert val_data['User', 'HAS_P_ACCESS', 'Resource'].num_edges == 535956 # 595506*0.9
        assert val_data['User', 'HAS_P_ACCESS', 'Resource'].edge_label_index.size(1) == 178650 # 595506*0.1 * 3 (2 (neg) + 1 (pos))
        assert val_data['Resource', 'rev_HAS_P_ACCESS', 'User'].num_edges == 535956 #
        # Negative edges with ratio 2:1:
        assert val_data['User', 'HAS_P_ACCESS', 'Resource'].edge_label.long().bincount().tolist() == [119100, 59550] # 595506*0.2 : 595506*0.1
        assert val_data['User', 'HAS_P_ACCESS', 'Resource'].edge_label.min() == 0
        assert val_data['User', 'HAS_P_ACCESS', 'Resource'].edge_label.max() == 1


        #%% Defining Mini-batch Loaders

        from torch_geometric.loader import LinkNeighborLoader

        # Define seed edges:
        edge_label_index = train_data['User', 'HAS_P_ACCESS', 'Resource'].edge_label_index
        edge_label = train_data['User', 'HAS_P_ACCESS', 'Resource'].edge_label


        train_loader = LinkNeighborLoader(
            data=train_data,
            num_neighbors=[20, 10],
            neg_sampling_ratio=2.0,
            edge_label_index=(('User', 'HAS_P_ACCESS', 'Resource'), edge_label_index),
            edge_label=edge_label,
            batch_size=batch_size,
            shuffle=True,
        )

        val_loader = LinkNeighborLoader(
            data=val_data,
            num_neighbors=[20, 10],
            edge_label_index=(('User', 'HAS_P_ACCESS', 'Resource'), edge_label_index),
            edge_label=edge_label,
            batch_size=3 * batch_size,
            shuffle=False,
        )

        # Inspect a sample:
        sampled_data = next(iter(train_loader))

        print("Sampled mini-batch:")
        print("===================")
        print(sampled_data)

        assert sampled_data['User', 'HAS_P_ACCESS', 'Resource'].edge_label_index.size(1) == 3 * batch_size # 2 negative + 1 positive
        assert sampled_data['User', 'HAS_P_ACCESS', 'Resource'].edge_label.min() == 0
        assert sampled_data['User', 'HAS_P_ACCESS', 'Resource'].edge_label.max() == 1



    #%% ## early stop train process
        if ablation_setting == 'Title':
            model = Title_Model(hidden_channels=hidden_channel)
        elif ablation_setting == 'Department':
            model = Department_Model(hidden_channels=hidden_channel)
        elif ablation_setting == 'Manager':
            model = Manager_Model(hidden_channels=hidden_channel)
        elif ablation_setting == 'All':
            model = Model(hidden_channels=hidden_channel)
        elif ablation_setting == 'None':
            model = None_Model(hidden_channels=hidden_channel)
        print(model)

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Device: '{device}'")
        model = model.to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

        best_model_path = os.path.join('./data/', 'early_stop_model')
        os.makedirs(best_model_path, exist_ok=True)
        best_model_path = os.path.join(best_model_path, setting+'_best.pt')
        metric = 'loss'
        early_stopping = EarlyStopping(save_path=best_model_path, verbose=(True), patience=10, delta=0.00001, metric=metric)

        train_loss_list = []
        validation_loss_list = []
        total_start_time = time.time()
        for epoch in np.arange(Epoch):
            start_time = time.time()
            train_loss, validation_loss = train(train_loader,val_loader)
            end_time = time.time()
            train_time = end_time - start_time
            print(f"Epoch: {epoch:03d}, train_time: {train_time} seconds, train_Loss: {train_loss:.4f},val_Loss: {validation_loss:.4f}")
            train_loss_list.append(train_loss)
            validation_loss_list.append(validation_loss)
            early_stopping(validation_loss, model)
            if early_stopping.early_stop:
                print("Early stopping at epoch:", epoch)
                break
        total_end_time = time.time()
        total_train_time = total_end_time - total_start_time
        print(f"Setting: {setting}, total_train_epoch:  {epoch:03d}, total_train_time: {total_train_time} seconds.")

        #save result:
        csv_file_path = "./data/Ablation_study_training_progress.csv"
        with open(csv_file_path, mode='a', newline='') as file:
            writer = csv.writer(file)
            writer.writerow([setting, "total_train_epoch:", epoch, 'total_train_time',total_train_time])
            writer.writerow([setting,'train_loss_list'])
            writer.writerow([setting,'validation_loss_list'])
            writer.writerow(train_loss_list)
            writer.writerow(validation_loss_list)

        # early_stopping.draw_trend(train_loss_list, validation_loss_list)
        train_list = train_loss_list
        test_list = validation_loss_list

        plt.plot(range(1, len(train_list) + 1), train_list, label='Training ' + metric)
        plt.plot(range(1, len(test_list) + 1), test_list, label='Validation ' + metric)

        # find position of check point,-1 means this a minimize problem like loss or cost
        if early_stopping.sign == -1:
            checkpoint = test_list.index(min(test_list)) + 1
        else:
            checkpoint = test_list.index(max(test_list)) + 1

        plt.axvline(checkpoint, linestyle='--', color='r', label='Early Stopping Checkpoint')

        plt.xlabel('epochs')
        plt.ylabel(metric)
        # plt.ylim(min(train_list + test_list), max(train_list + test_list))  # consistent scale
        # plt.xlim(0, len(test_list) + 1)  # consistent scale
        plt.grid(True)
        plt.legend()
        plt.tight_layout()
        plt.savefig('./data/'+setting+'Traning_Eearly_Stop', dpi=600)
        plt.show()
