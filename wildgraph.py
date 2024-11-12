import torch
from torch import optim, nn
from torch.utils.data import Dataset
from torch_geometric.nn import Node2Vec

import lightning as L

import pdb
import random
from util import dist, heatmap


# define any number of nn.Modules (or use your current ones)
EMBEDDING_DIM = 128
GMM_DIM = 3
DEVICES = 1
START_NODE_LABEL = -1
HIDDEN_DIM = 600
EMBED_EPOCHS = 70

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
    def __init__(self, embed_model, file_name, transitions, tokenizer, embed_dim, hidden_dim, gmm_dim, vocab_dim, seq_length, dropout_rate=0.1, pe=1):
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
                nn.Linear(self.hparams.embed_dim + self.hparams.gmm_dim + self.hparams.pe, self.hparams.hidden_dim//2), nn.ReLU(), 
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
        if (self.hparams.pe == 1):
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
        Y = 5
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
                if (self.hparams.pe == 1):
                    x_encoded = torch.concat([z.squeeze(), x.squeeze(), pos], dim=0)
                else:
                    x_encoded = torch.concat([z.squeeze(), x.squeeze()], dim=0)
                y = self.model['decoder'](x_encoded)
                index = torch.argmax(y)
                #pdb.set_trace()
                topk = torch.topk(y, Y)
                xs = self.embed_model(topk.indices)
                x = torch.matmul(F.softmax(topk.values, dim=0), xs)
                
                traj.append(int(index.cpu().numpy()))
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

def train_wildgraph(data, embedder, tokenizer, train_loader, validation_loader, epochs, file_name, seq_length, embed_dim = -1, pe = 1):
#def train_wildgraph(data, embedder, tokenizer, train_loader, epochs, file_name):
    #wildgraph = WildGraph.load_from_checkpoint('lightning_logs/version_603/checkpoints/epoch=66-step=46431.ckpt', embed_model=embedder, transitions = data.transitions, embed_dim = EMBEDDING_DIM, hidden_dim = HIDDEN_DIM, gmm_dim= GMM_DIM, vocab_dim = data.vocab_dim, seq_length=186)
    #return wildgraph
    if embed_dim < 0:
        embed_dim = EMBEDDING_DIM
    wildgraph = WildGraph(file_name=file_name, embed_model=embedder, tokenizer=tokenizer, transitions = data.transitions, embed_dim = embed_dim, hidden_dim = HIDDEN_DIM, gmm_dim= GMM_DIM, vocab_dim = data.vocab_dim, seq_length=seq_length, pe=pe)
    trainer = L.Trainer(max_epochs=epochs, devices=DEVICES) #, callbacks=[early_stop_callback])
    trainer.fit(wildgraph, train_loader, validation_loader)
    wildgraph.eval()
    return wildgraph
