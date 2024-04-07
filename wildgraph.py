from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
from sklearn.model_selection import KFold
from torch import optim, nn, utils, Tensor, rand
import torch
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader


import numpy as np
from torch_geometric.nn import Node2Vec
from sklearn.metrics.pairwise import cosine_similarity
import random
from torch_geometric.nn import Node2Vec
from util import dist, dataset, heatmap
import pandas as pd

# define any number of nn.Modules (or use your current ones)
EMBEDDING_DIM = 128
GMM_DIM = 3
DEVICES = 1
START_NODE_LABEL = -1
HIDDEN_DIM = 600
EMBED_EPOCHS = 150
PE = 1

seed = torch.Generator().manual_seed(42)

class RawDataset(Dataset):
    def __init__(self, df):
        self.trajs = df
    def __len__(self):
        return len(self.trajs)
    def __getitem__(self, index):
        return self.trajs[index]

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
    def __init__(self, embed_model, file_name, transitions, tokenizer, embed_dim, hidden_dim, gmm_dim, vocab_dim, seq_length, dropout_rate=0.1):
        super().__init__()
        self.save_hyperparameters(ignore=["embed_model", "transitions", "tokenizer"])
        self._create_model()
        self.embed_model = embed_model
        self.transitions = transitions
        self.hausdorff = dist.Hausdorff()
        self.dtw = dist.DTW()
        self.fde = dist.FDE()
        self.tokenizer = tokenizer #joblib.load('./label_encoder.0.5.joblib')
        self.heatmap = heatmap.Heatmap(file_name)

    def _create_model(self):
        self.model = nn.ModuleDict({
            'encoder': nn.Sequential(
                nn.Linear(self.hparams.embed_dim, self.hparams.hidden_dim), nn.ReLU(), #nn.Dropout(0.1), 
                nn.Linear(self.hparams.hidden_dim, self.hparams.hidden_dim//2), nn.ReLU(), 
                nn.Linear(self.hparams.hidden_dim//2, self.hparams.gmm_dim)
            ),
            'decoder': nn.Sequential(
                nn.Linear(self.hparams.embed_dim + self.hparams.gmm_dim + PE, self.hparams.hidden_dim//2), nn.ReLU(), 
                nn.Linear(self.hparams.hidden_dim//2, self.hparams.hidden_dim), nn.ReLU(), 
                nn.Linear(self.hparams.hidden_dim, self.hparams.vocab_dim)
            )
        })

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
        return 
        z_samples, z_init = self.process()
        real = data
        #pdb.set_trace()
        generated = self.generate_traj(z_samples, z_init, 12)
        #generated = [g.cpu().numpy() for g in generated]
        
        min, max, mean = self.hausdorff.update(real.cpu(), generated)
        self.log("hau(min)", min, prog_bar=True, on_step=False, on_epoch=True)
        self.log("hau(max)", max, prog_bar=True, on_step=False, on_epoch=True)
        self.log("hau(mean)", mean, prog_bar=True, on_step=False, on_epoch=True)

        min, max, mean = self.dtw.update(real.cpu(), generated)
        self.log("dtw(min)", min, prog_bar=True, on_step=False, on_epoch=True)
        self.log("dtw(max)", max, prog_bar=True, on_step=False, on_epoch=True)
        self.log("dtw(mean)", mean, prog_bar=True, on_step=False, on_epoch=True)

        min, max, mean = self.fde.update(real.cpu(), generated)
        self.log("fde(min)", min, prog_bar=True, on_step=False, on_epoch=True)
        self.log("fde(max)", max, prog_bar=True, on_step=False, on_epoch=True)
        self.log("fde(mean)", mean, prog_bar=True, on_step=False, on_epoch=True)
    def forward(self, x, y, pos):
        #return self.positional_encoding(self.embed_model(x.long()).squeeze(1), pos)
        z = self.model['encoder'](y.squeeze(-1))
        if (PE == 1):
            encoded = torch.concat([z, x, pos.unsqueeze(1)], dim=1)
        else:
            encoded = torch.concat([z, x], dim=1)
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
            x = self.embed_model(torch.tensor([i]))
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
            x = self.embed_model([-1])
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
                if (PE == 1):
                    x_encoded = torch.concat([z.squeeze(), x.squeeze(), pos], dim=0)
                else:
                    x_encoded = torch.concat([z.squeeze(), x.squeeze()], dim=0)
                y = self.model['decoder'](x_encoded)
                index = torch.argmax(y)
                traj.append(int(index.cpu().numpy()))
                x = self.embed_model([index])
                z_array = z_samples[index]

            if (not skip):
                cells = self.tokenizer.inverse_transform(traj)
                traj_long_lat = [self.heatmap.sample(cell) for cell in cells]
                #pdb.set_trace()
                trajs.append(traj_long_lat)
            #else:
            #    i = i - 1
        return trajs


    #Simple Embedder
def embed(x, vocab_dim):
    x_embed = -1 * torch.ones((vocab_dim, 1)).squeeze(-1)
    if (x < 0):
        return x_embed
    x_embed[x] = 4
    return x_embed

def train_embed(edge_index, embed_dim = EMBEDDING_DIM,  epochs = EMBED_EPOCHS, devices = DEVICES):
    #embed_model = EmbeddingModel.load_from_checkpoint('lightning_logs/version_611/checkpoints/epoch=159-step=2080.ckpt')
    #return embed_model
    embed_model = EmbeddingModel(embed_dim = embed_dim, edge_index=edge_index)
    embedTrainer = L.Trainer(max_epochs=epochs, devices=devices, reload_dataloaders_every_n_epochs=1)
    embedTrainer.fit(embed_model)
    return embed_model

def train_wildgraph(data, embedder, tokenizer, train_loader, validation_loader, epochs, file_name, seq_length):
#def train_wildgraph(data, embedder, tokenizer, train_loader, epochs, file_name):
    #wildgraph = WildGraph.load_from_checkpoint('lightning_logs/version_603/checkpoints/epoch=66-step=46431.ckpt', embed_model=embedder, transitions = data.transitions, embed_dim = EMBEDDING_DIM, hidden_dim = HIDDEN_DIM, gmm_dim= GMM_DIM, vocab_dim = data.vocab_dim, seq_length=186)
    #return wildgraph
    wildgraph = WildGraph(file_name=file_name, embed_model=embedder, tokenizer=tokenizer, transitions = data.transitions, embed_dim = EMBEDDING_DIM, hidden_dim = HIDDEN_DIM, gmm_dim= GMM_DIM, vocab_dim = data.vocab_dim, seq_length=seq_length)
    trainer = L.Trainer(max_epochs=epochs, devices=DEVICES) #, callbacks=[early_stop_callback])
    trainer.fit(wildgraph, train_loader, validation_loader)
    wildgraph.eval()
    return wildgraph
"""
file_path = 'trajs.csv'
columns = ['idx', 'label', 'location.lat', 'location.long']
df = pd.read_csv(file_path, usecols=columns)
fold_count = 5
kf= KFold(n_splits=fold_count)
X = range(60)
SAMPLES_TEST = 60

import datetime

import logging
#logger = logging.getLogger('wildgrpah_application')

#current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logging.basicConfig(filename=f"wildgraph_{EMBED_EPOCHS}_{RNN_EPOCHS}.log", level=logging.INFO)
for rep in range(5):
    min1_c, max1_c, mean1_c, touched1_c = 0, 0, 0, 0 
    min2_c, max2_c, mean2_c, touched2_c = 0, 0, 0, 0 
    min3_c, max3_c, mean3_c, touched3_c = 0, 0, 0, 0 
    logging.info(f"EXPERIMENT ${rep}: ####################################")
    for k, (train_index, test_index) in enumerate(kf.split(X)):
        df_train = df[df['label'].isin(train_index)].copy()
        data = dataset.TrajDataset(df_train, len(train_index), seq_length=185)
        tokenizer = data.tokenizer

        df_test = df[df['label'].isin(test_index)]
        real = df[['label', 'location.lat', 'location.long']].groupby('label')
        group_arrays = [group[['location.lat', 'location.long']].to_numpy() for _, group in real]
        real_array = np.array(group_arrays)
        test_set = RawDataset(real_array)

        # Split the data into training and testing sets
        X_train, X_test = data, test_set
        train_loader = DataLoader(X_train, batch_size=16, num_workers=8)
        validation_loader = DataLoader(X_test, batch_size=16, num_workers=8)


        embedder = train_embed(data.edge_index)
        wildgraph = WildGraph(embed_model=embedder, tokenizer=tokenizer, transitions = data.transitions, embed_dim = EMBEDDING_DIM, hidden_dim = HIDDEN_DIM, gmm_dim= GMM_DIM, vocab_dim = data.vocab_dim, seq_length=185)
        wildgraph = train_wildgraph(data, embedder, tokenizer, train_loader, validation_loader)
        wildgraph.eval()
        z_samples, z_init = wildgraph.process()
        generated = wildgraph.generate_traj(z_samples, z_init, SAMPLES_TEST)
        min1, max1, mean1, touched1 = wildgraph.hausdorff.update(generated, real_array)
        min2, max2, mean2, touched2 = wildgraph.dtw.update(generated, real_array)
        min3, max3, mean3, touched3 = wildgraph.fde.update(generated, real_array)
        np.save(f"./wild_generated_samples/wildgraph_{rep}_k{k}_EPOCHS={RNN_EPOCHS}.npy", generated)

        logging.info({"k": k,"########### Means:": "",  "mean_haus": mean1, "mean_dtw": mean2, "mean_fde": mean3})
        #logging.info({"  ":"  ", "min_h": min1, "max_h": max1, "mean_haus": mean1, "min_d": min2, "max_d": max2, "mean_dtw": mean2, "min_f": min3, "max_f": max3, "mean_f": mean3})

        #logging.info("K1 ", min1, max1, mean1, min2, max2, mean2, min3, max3, mean3)
        min1_c, max1_c, mean1_c, touched1_c = min1 + min1_c, max1 + max1_c, mean1_c + mean1, touched1_c + touched1
        min2_c, max2_c, mean2_c, touched2_c = min2 + min2_c, max2 + max2_c, mean2_c + mean2, touched2_c + touched2
        min3_c, max3_c, mean3_c, touched3_c = min3 + min3_c, max3 + max3_c, mean3_c + mean3, touched3_c + touched3
    
    res = np.array([min1_c, max1_c, mean1_c, min2_c, max2_c, mean2_c, min3_c, max3_c, mean3_c]) / (fold_count)
    logging.info(f"END Experiment {rep} FINAL: Means: (mean_haus, mean_dtw, mean_fde) {res[2]}, {res[5]}, {res[8]}, ({touched1_c}, {touched1_c}, {touched1_c})")
    logging.info({"  ":"  ", "k": "FINAL", "min_h": res[0], "max_h": res[1], "mean_haus": res[2], "min_d": res[3], "max_d": res[4], "mean_dtw": res[5], "min_f": res[6], "max_f": res[7], "mean_f": res[8]})
"""