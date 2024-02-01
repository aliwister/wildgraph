from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from torch import optim, nn, utils, Tensor, rand
import torch
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader

from sklearn.mixture import GaussianMixture
import scipy
import math
import numpy as np
import pdb
from torch_geometric.nn import Node2Vec
import joblib
from sklearn.metrics.pairwise import cosine_similarity
import random
from torch_geometric.nn import Node2Vec


# define any number of nn.Modules (or use your current ones)
EMBEDDING_DIM = 128
GMM_DIM = 3
DEVICES = 1
START_NODE_LABEL = -1
HIDDEN_DIM = 600
EMBED_EPOCHS = 110
RNN_EPOCHS = 31
seed = torch.Generator().manual_seed(42)

import h3
from shapely.geometry import Polygon, Point
def softmax(x): 
    np.exp(x)/sum(np.exp(x))
class Heatmap():
    def __init__(self, file):
        self.heatmap_resolution = 7
        self.generate_heatmap(file)
    
    def generate_heatmap(self, file):
        #data = joblib.load('heatmap.0.5.joblib')
        #dict = {row[0]: row[1] for row in data}
        columns = ['idx', 'label', 'location.lat', 'location.long']
        df = pd.read_csv(file, usecols=columns)
        heatmap_resolution = 6
        df['heatmap'] = df.apply(lambda row: h3.latlng_to_cell(row['location.lat'], row['location.long'], heatmap_resolution), axis=1)
        heatmap_counts = df['heatmap'].value_counts().reset_index()
        heatmap_counts.columns = ['heatmap', 'occurrences']
        self.dict = {row[0]: row[1] for row in heatmap_counts.values}

    def sample(self, h3_region_id):

        cells = np.array(list(h3.cell_to_children(h3_region_id, self.heatmap_resolution))) 
        values = [self.dict.get(key, 0) for key in cells]
        selected_cell = np.random.choice(cells, p=softmax(values))      

        # Get the polygon vertices of the H3 region
        hexagon_vertices = h3.cell_to_boundary(str(selected_cell), geo_json=False)
        
        # Create a Shapely Polygon from the vertices
        hexagon_polygon = Polygon(hexagon_vertices)
        
        # Generate a random point inside the polygon
        min_x, min_y, max_x, max_y = hexagon_polygon.bounds
        while True:
            random_point = Point(random.uniform(min_x, max_x), random.uniform(min_y, max_y))
            if hexagon_polygon.contains(random_point):
                break
        
        return [random_point.x, random_point.y]

from scipy.spatial.distance import directed_hausdorff
import pandas as pd
class Hausdorff():
    def __init__(self) -> None:
        df = pd.read_csv('trajs.csv')
        real = df[['label', 'location.lat', 'location.long']].groupby('label')
        real.head()
        group_arrays = [group[['location.lat', 'location.long']].to_numpy() for _, group in real]
        self.real_array = np.array(group_arrays)
         
    def update(self, generated):
        hausdorff_distances = []

        for traj1 in generated:
            path = 10000
            skip = False
            for traj2 in self.real_array:
                #pdb.set_trace()
                try:
                    if(len(traj1) != len(traj2)):
                        #print ("Skipping Short Traj")
                        skip = True
                        break
                except TypeError:
                    print("Type Error", traj1, generated)
                    return 0,0,0
                distance1 = directed_hausdorff(traj1, traj2)[0]
                distance2 = directed_hausdorff(traj2, traj1)[0]
                distance = max(distance1, distance2)
                if distance < path:
                    path = distance
            hausdorff_distances.append(path)

        if (not skip):
            min_hausdorff_distance = np.min(hausdorff_distances)
            max_hausdorff_distance = np.max(hausdorff_distances)
            mean_hausdorff_distance = np.mean(hausdorff_distances)
            return min_hausdorff_distance, max_hausdorff_distance, mean_hausdorff_distance
        return -1, -1, -1

from sklearn.preprocessing import LabelEncoder
import geopandas as gpd
from scipy.spatial.distance import pdist
class TrajDataset(Dataset):
    def __init__(self, file):
        self.tokenizer = LabelEncoder()
        x, y, pos = self.load_data(file)
        self.x = torch.tensor(x, dtype=torch.long)
        self.y = torch.tensor(y, dtype=torch.long)
        self.pos = torch.tensor(pos, dtype=torch.long)
        
    def __len__(self):
        return len(self.x)
    def __getitem__(self, index):
        return self.x[index], self.y[index], self.pos[index]
    def load_data(self, file):
        init_resolution = 2
        file_path = file  # Replace this with the actual path to your CSV file
        columns = ['idx', 'label', 'location.lat', 'location.long']
        df = pd.read_csv(file_path, usecols=columns)
        df['h3_address'] = df.apply(lambda row: h3.latlng_to_cell(row['location.lat'], row['location.long'], init_resolution), axis=1)

        def get_distance_split_targets(dfi, dist): 
            value_counts = dfi['h3_address'].value_counts()
            dup_list = value_counts[value_counts > 1].index.tolist()
            dup_list_filtered = []
            for h3 in dup_list:
                dfii = dfi[dfi['h3_address']==h3]
                distances = pdist(dfii[['location.lat', 'location.long']], metric='euclidean')
                d = max(distances)
                if (d > dist):
                    dup_list_filtered.append(h3)
            return dup_list_filtered

        def granulate(row, resolution, dup_list):
            # Example: Multiply 'other_column' by 2 for rows where 'h3_address' is in h3_10
            if row['h3_address'] in dup_list:
                return h3.latlng_to_cell(row['location.lat'], row['location.long'], resolution)
            else:
                return row['h3_address']

        for zoom in range(3,8):
            dup_list = get_distance_split_targets(df, .25)
            df['h3_address'] = df.apply(lambda row: granulate(row, zoom, dup_list), axis=1)

        df = df.sort_values(by=['idx', 'label'])
        unique_h3_codes = df['h3_address'].unique()
        nodes = self.tokenizer.fit_transform(unique_h3_codes)
        df['node'] = self.tokenizer.fit_transform(df['h3_address'])
        export = df['node'].values.reshape(60,185)
        export1 = np.c_[np.ones(export.shape[0])*-1, export]
        datafile = export1
        edges_set = set()
        transitions = []
        X, y, pos = [], [], []
        #file = scipy.io.loadmat(file)
        #datafile = file['data'].astype(np.int64)
        for sequence in datafile:
            for j in range(len(sequence) - 1):
                if (sequence[j] != sequence[j+1] ):
                    edges_set.add((sequence[j], sequence[j+1]))
                X.append(sequence[j]) 
                y.append(sequence[j+1])
                pos.append(j)
                transitions.append((sequence[j], sequence[j+1]))
        self.vocab_dim = int(max(y)+1)
        edges_array = np.array(list(edges_set))
        self.edge_index = torch.tensor(edges_array.T, dtype=torch.long)
        self.transitions = transitions
        return X, y, pos
    #@property
    #def edge_index(self):
    #    return self.edge_index
    #@property
    #def transitions(self):
    #    return self.transitions
    #@property
    #def vocab_dim(self):
    #    return self.vocab_dim

class EmbeddingModel(L.LightningModule):
    def __init__(self, embed_dim, edge_index):
        super().__init__()
        self.save_hyperparameters()
        # Define an embedding layer to be trained
        self.embedding_layer = Node2Vec(
            edge_index,
            embedding_dim=embed_dim,
            walks_per_node=1,
            walk_length=20,
            context_size=10,
            p=1.0,
            q=1.0,
            num_negative_samples=1,
        )

    def training_step(self, batch, batch_idx):
        pos_rw, neg_rw = batch
        loss = self.embedding_layer.loss(pos_rw, neg_rw)
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def train_dataloader(self):
        return self.embedding_layer.loader(batch_size=128, shuffle=True, num_workers=4)

    def forward(self, x):
        return self.embedding_layer(x)

    def configure_optimizers(self):
        # Define and return an optimizer for training the embedding layer
        optimizer = torch.optim.Adam(self.parameters(), lr=0.01)
        return optimizer

# define the LightningModule
class WildGraph(L.LightningModule):
    def __init__(self, embed_model, transitions, tokenizer, embed_dim, hidden_dim, gmm_dim, vocab_dim, seq_length, dropout_rate=0.1):
        super().__init__()
        self.save_hyperparameters(ignore=["embed_model", "transitions", "tokenizer"])
        self._create_model()
        self.embed_model = embed_model
        self.transitions = transitions
        self.hausdorff = Hausdorff()
        self.tokenizer = tokenizer #joblib.load('./label_encoder.0.5.joblib')
        self.heatmap = Heatmap('trajs.csv')

    def _create_model(self):
        self.model = nn.ModuleDict({
            'encoder': nn.Sequential(
                nn.Linear(self.hparams.embed_dim, self.hparams.hidden_dim), nn.ReLU(), #nn.Dropout(0.1), 
                nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim//2), nn.ReLU(), 
                nn.Linear(self.hparams.hidden_dim//2, self.hparams.gmm_dim)
            ),
            'decoder': nn.Sequential(
                nn.Linear(self.hparams.embed_dim + self.hparams.gmm_dim + 1, self.hparams.hidden_dim//2), nn.ReLU(), 
                nn.Linear(self.hparams.hidden_dim//2, self.hparams.hidden_dim), nn.ReLU(), 
                nn.Linear(self.hparams.hidden_dim, self.hparams.vocab_dim)
            )
        })

        #self.loss = nn.CrossEntropyLoss()
        self.loss = nn.CrossEntropyLoss()
        #self.positional_encoding = PositionalEncoding(self.hparams.embed_dim, self.hparams.seq_length)
        #self.embedding = nn.Embedding(self.hparams.vocab_dim, self.hparams.embed_dim)
        #self.layer_norm = nn.LayerNorm(self.hparams.embed_dim)

    def training_step(self, batch, batch_idx):
        x, y, pos = batch
        embed_x = self.embed_model(x)
        embed_y = self.embed_model(y)
        
        y_hat = self(embed_x, embed_y, pos)
        #pdb.set_trace()
        loss = self.loss(y_hat, y.long())

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, data, batch_idx):
        z_samples, z_init = self.process()
        #pdb.set_trace()
        generated = self.generate_traj(z_samples, z_init, 10)
        
        min, max, mean = self.hausdorff.update(generated)
        self.log("hausdorff (min)", min, prog_bar=True, on_step=False, on_epoch=True)
        self.log("hausdorff (max)", max, prog_bar=True, on_step=False, on_epoch=True)
        self.log("hausdorff (mean)", mean, prog_bar=True, on_step=False, on_epoch=True)

    def forward(self, x, y, pos):
        #return self.positional_encoding(self.embed_model(x.long()).squeeze(1), pos)
        z = self.model['encoder'](y.squeeze(-1))
        encoded = torch.concat([z, x, pos.unsqueeze(1)], dim=1)
        y_hat = self.model['decoder'](encoded)
        return y_hat

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer

    def process(self):
        z_s = []
        # Evaluate z's for each state. z[0] -> state 1
        #pdb.set_trace()
        for i in range(self.hparams.vocab_dim):
            x = embedder(torch.tensor(i))
            z = self.model['encoder'](x)
            z_s.append(z)

        # Evaluate z sample for each state. z_samples[0] -> state 0. z_samples[1] -> state 1
        z_samples = [ [] for _ in range(self.hparams.vocab_dim) ]
        z_init = []
        for i in self.transitions:
            idx = int(i[0])
            target = int(i[1])
            if (idx == -1): #For state -1
                z_init.append(z_s[target])
            else:
                z_samples[idx].append(z_s[target])

        return z_samples, z_init
    
        # Generate New Trajectories. State at state -1
    def generate_traj(self, z_samples, z_init, num):
        trajs = []
        for i in range(num):
            #pdb.set_trace()
            x = self.embed_model(-1)
            traj = []
            index = 0
            z_array = z_init
            skip = False
            
            for i in range(self.hparams.seq_length):
                pos = torch.tensor([i])
                pos = pos.type_as(x)
                if (len(z_array) == 0):
                    print ("early stop")
                    skip = True
                    break
                    #return traj
                z = random.sample(z_array, 1)[0]
                #pdb.set_trace()
                x_encoded = torch.concat([z, x, pos], dim=0)
                y = self.model['decoder'](x_encoded)
                index = torch.argmax(y)
                traj.append(int(index.cpu().numpy()))
                x = self.embed_model(index)
                z_array = z_samples[index]

            if (not skip):
                cells = self.tokenizer.inverse_transform(traj)
                traj_long_lat = [self.heatmap.sample(cell) for cell in cells]
                #pdb.set_trace()
                trajs.append(traj_long_lat)
            else:
                i = i - 1
        return trajs

#Simple Embedder
def embed(x, vocab_dim):
    x_embed = -1 * torch.ones((vocab_dim, 1)).squeeze(-1)
    if (x < 0):
        return x_embed
    x_embed[x] = 4
    return x_embed

def train_embed(edge_index, embed_dim = EMBEDDING_DIM,  epochs = EMBED_EPOCHS, devices = DEVICES):
    embed_model = EmbeddingModel.load_from_checkpoint('lightning_logs/version_611/checkpoints/epoch=159-step=2080.ckpt')
    return embed_model
    embed_model = EmbeddingModel(embed_dim = embed_dim, edge_index=edge_index)
    embedTrainer = L.Trainer(max_epochs=epochs, devices=devices, reload_dataloaders_every_n_epochs=1)
    embedTrainer.fit(embed_model)
    return embed_model

def train_wildgraph(dataset, embedder, tokenizer):
    #wildgraph = WildGraph.load_from_checkpoint('lightning_logs/version_603/checkpoints/epoch=66-step=46431.ckpt', embed_model=embedder, transitions = dataset.transitions, embed_dim = EMBEDDING_DIM, hidden_dim = HIDDEN_DIM, gmm_dim= GMM_DIM, vocab_dim = dataset.vocab_dim, seq_length=186)
    #return wildgraph
    wildgraph = WildGraph(embed_model=embedder, tokenizer=tokenizer, transitions = dataset.transitions, embed_dim = EMBEDDING_DIM, hidden_dim = HIDDEN_DIM, gmm_dim= GMM_DIM, vocab_dim = dataset.vocab_dim, seq_length=185)
    train_set_size = int(len(dataset) * .999)
    valid_set_size = len(dataset) - train_set_size

    # split the train set into two
    train_set, valid_set = data.random_split(dataset, [train_set_size, valid_set_size], generator=seed)
    train_loader = DataLoader(train_set, batch_size=16, num_workers=8)
    validation_loader = DataLoader(valid_set, batch_size=16, num_workers=8)

    trainer = L.Trainer(max_epochs=RNN_EPOCHS, devices=DEVICES) #, callbacks=[early_stop_callback])
    trainer.fit(wildgraph, train_loader, validation_loader)
    wildgraph.eval()
    return wildgraph

file = 'trajs.csv'
dataset = TrajDataset(file)
tokenizer = dataset.tokenizer

embedder = train_embed(dataset.edge_index)
wildgraph = train_wildgraph(dataset, embedder, tokenizer)

# Load the model from the checkpoint
#autoencoder = LitAutoEncoder.load_from_checkpoint('lightning_logs/version_135/checkpoints/epoch=99-step=888000.ckpt')

z_samples, z_init = wildgraph.process()
generated = wildgraph.generate_traj(z_samples, z_init, 60)
#print(generated)
joblib.dump(generated, 'generated.npy')
hausdorff = Hausdorff()
min, max, mean = hausdorff.update(generated)
print ("hausdorff (min)", min, "hausdorff (max)", max, "hausdorff (mean)", mean)
