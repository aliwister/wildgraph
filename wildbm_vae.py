import os
from sklearn.model_selection import KFold
from torch import optim, nn, utils, Tensor, rand
import torch

from torchvision.transforms import ToTensor
from torchvision import datasets
import torchvision.transforms as transforms
import lightning as L
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import DataLoader
from torch.utils.data import Dataset, DataLoader

from sklearn.mixture import GaussianMixture

from util import dist
import pandas as pd
import numpy as np
import logging


class RawDataset(Dataset):
    def __init__(self, df):
        self.trajs = df
    def __len__(self):
        return len(self.trajs)
    def __getitem__(self, index):
        return self.trajs[index]

class CustomDataset(Dataset):
    def __init__(self, df, tokenizer=None):
        self.data = df
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        #print(self.data[idx])
        return torch.tensor(self.data[idx]).float()
    

# define the LightningModule
class LitAutoEncoder(L.LightningModule):
    def __init__(self, seq_length):
        super().__init__()
        self.encoder = nn.Sequential(nn.Linear(seq_length*2, 64), nn.ReLU(), nn.Linear(64, 3))
        self.decoder = nn.Sequential(nn.Linear(3, 64), nn.ReLU(), nn.Linear(64, seq_length*2))
        self.gmm = GaussianMixture(n_components=3)
        self.hausdorff = dist.Hausdorff()
        self.dtw = dist.DTW()
        self.fde = dist.FDE()

    def training_step(self, batch, batch_idx):
        # training_step defines the train loop.
        # it is independent of forward
        x = batch
        x = x.view(x.size(0), -1)
        z = self.encoder(x)
        x_hat = self.decoder(z)
        loss = nn.functional.mse_loss(x_hat, x)

        # Logging to TensorBoard (if installed) by default
        self.log("train_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        # this is the validation loop
        #real = data
        #encoded_data = self.encoder(real).detach().cpu().numpy()
        #self.gmm.fit(encoded_data)
        #pdb.set_trace()
        #random_samples = self.gmm.sample(12)[0]
        #generated = self.decoder(random_samples)

        #generated = [g.cpu().numpy() for g in generated]
        
        #min, max, mean = self.hausdorff.update(real.cpu(), generated)
        #self.log("hau(min)", min, prog_bar=True, on_step=False, on_epoch=True)
        #self.log("hau(max)", max, prog_bar=True, on_step=False, on_epoch=True)
        #self.log("hau(mean)", mean, prog_bar=True, on_step=False, on_epoch=True)

        #min, max, mean = self.dtw.update(real.cpu(), generated)
        #self.log("dtw(min)", min, prog_bar=True, on_step=False, on_epoch=True)
        #self.log("dtw(max)", max, prog_bar=True, on_step=False, on_epoch=True)
        #self.log("dtw(mean)", mean, prog_bar=True, on_step=False, on_epoch=True)

        #min, max, mean = self.fde.update(real.cpu(), generated)
        #self.log("fde(min)", min, prog_bar=True, on_step=False, on_epoch=True)
        #self.log("fde(max)", max, prog_bar=True, on_step=False, on_epoch=True)
        #self.log("fde(mean)", mean, prog_bar=True, on_step=False, on_epoch=True)
        print("Validation")

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        return optimizer


# init the autoencoder


"""
# download the MNIST datasets
file_path = 'trajs.csv'
columns = ['idx', 'label', 'location.lat', 'location.long']
df = pd.read_csv(file_path, usecols=columns)

fold_count = 5
kf= KFold(n_splits=fold_count)
X = range(60)

logging.basicConfig(filename='wildlog_VAE.log', level=logging.INFO)

#fh = logging.FileHandler(f"{current_datetime}.log")
#fh.setLevel(logging.INFO)
#formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
#fh.setFormatter(formatter)
#logger.addHandler(fh)

#filename = f"{current_datetime}.txt"  # Or any other extension you want
SAMPLES_TEST = 60

for rep in range(5):
    min1_c, max1_c, mean1_c = 0, 0, 0 
    min2_c, max2_c, mean2_c = 0, 0, 0 
    min3_c, max3_c, mean3_c = 0, 0, 0 
    logging.info(f"EXPERIMENT ${rep}: ####################################")
    for k, (train_index, test_index) in enumerate(kf.split(X)):
        df_train = df[df['label'].isin(train_index)].copy()
        real = df_train[['label', 'location.lat', 'location.long']].groupby('label')
        group_arrays = [group[['location.lat', 'location.long']].to_numpy() for _, group in real]
        train_array = np.array(group_arrays)
        dataset = CustomDataset(train_array)

        df_test = df[df['label'].isin(test_index)]
        real = df_test[['label', 'location.lat', 'location.long']].groupby('label')
        group_arrays = [group[['location.lat', 'location.long']].to_numpy() for _, group in real]
        test_array = np.array(group_arrays)
        test_set = RawDataset(test_array)
        X_train, X_test = dataset, test_set

        train_loader = DataLoader(X_train, batch_size=8, num_workers=8)
        validation_loader = DataLoader(X_test, batch_size=8, num_workers=8)
        autoencoder = LitAutoEncoder()

        trainer = L.Trainer(max_epochs=100, devices=1)
        trainer.fit(autoencoder, train_loader)
        autoencoder.encoder.eval()

        x = torch.tensor(train_array)
        x = x.view(x.size(0), -1)
        # Fit GMM on encoded data
        encoded_data = autoencoder.encoder(x.float()).detach().cpu().numpy()
        autoencoder.gmm.fit(encoded_data)

        # Generate random samples from the GMM
        random_samples = autoencoder.gmm.sample(SAMPLES_TEST)[0]
        # Convert NumPy array to PyTorch tensor
        random_samples = torch.from_numpy(random_samples).float()

        # Decode the samples
        generated = autoencoder.decoder(random_samples).detach().cpu().numpy()
        generated = generated.reshape(SAMPLES_TEST, 185, 2)

        min1, max1, mean1 = autoencoder.hausdorff.update(generated, train_array)
        min2, max2, mean2 = autoencoder.dtw.update(generated, train_array)
        min3, max3, mean3 = autoencoder.fde.update(generated, train_array)

        logging.info({"Means:": "",  "mean_haus": mean1, "mean_dtw": mean2, "mean_fde": mean3})
        #logging.info({"k": k, "min_h": min1, "max_h": max1, "mean_haus": mean1, "min_d": min2, "max_d": max2, "mean_dtw": mean2, "min_f": min3, "max_f": max3, "mean_f": mean3})
        #logging.info("K1 ", min1, max1, mean1, min2, max2, mean2, min3, max3, mean3)
        min1_c, max1_c, mean1_c = min1 + min1_c, max1 + max1_c, mean1_c + mean1
        min2_c, max2_c, mean2_c = min2 + min2_c, max2 + max2_c, mean2_c + mean2
        min3_c, max3_c, mean3_c = min3 + min3_c, max3 + max3_c, mean3_c + mean3
    
    res = np.array([min1_c, max1_c, mean1_c, min2_c, max2_c, mean2_c, min3_c, max3_c, mean3_c]) / (fold_count)
    logging.info(f"END Experiment {rep} FINAL: Means: (mean_haus, mean_dtw, mean_fde) {res[2]}, {res[5]}, {res[8]}")
    logging.info({"  ":"  ", "k": "FINAL", "min_h": res[0], "max_h": res[1], "mean_haus": res[2], "min_d": res[3], "max_d": res[4], "mean_dtw": res[5], "min_f": res[6], "max_f": res[7], "mean_f": res[8]})
"""