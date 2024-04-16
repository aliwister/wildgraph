from argparse import ArgumentParser
import time
from sklearn.model_selection import KFold
import torch
import lightning as L
from torch.utils.data import DataLoader
import torch.nn.functional as F
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import numpy as np
from util import dist, dataset, heatmap
import pandas as pd

import logging
import wildgraph as wildgraph
import wildbm_transformers as transformer
import wildbm_vae as vae
import wildbm_gan as gan
import wildbm_wildgen as wildgen
import scipy

ABLATE_UNIFORM_COARSE = False
ABLATE_UNIFORM_FINE = False
ABLATE_NO_PE = False
ABLATE_BOW = False


seed = torch.Generator().manual_seed(42)
class RawDataset(Dataset):
    def __init__(self, df):
        self.trajs = df
    def __len__(self):
        return len(self.trajs)
    def __getitem__(self, index):
        return self.trajs[index]

class Runner():
    def __init__(self, seq_len, distance, test_sample_size):
        self.seq_len = seq_len
        self.distance = distance 
        self.test_sample_size = test_sample_size

    def run_wildgraph(self, df_train, df_test, epochs):
        data = dataset.TrajDataset(df_train, len(train_index), seq_length=self.seq_len, distance=self.distance)
        if ABLATE_UNIFORM_COARSE:
            data = dataset.TrajDataset(df_train, len(train_index), seq_length=self.seq_len, distance=self.distance, ablate_is_fixed_zoom=True, fixed_zoom_value=2)
        elif ABLATE_UNIFORM_FINE:
            data = dataset.TrajDataset(df_train, len(train_index), seq_length=self.seq_len, distance=self.distance, ablate_is_fixed_zoom=True, fixed_zoom_value=8)

        tokenizer = data.tokenizer
        #logging.info(f"Network Density = {data.density}")
        real = df_test[['label', 'location.lat', 'location.long']].groupby('label')
        group_arrays = [group[['location.lat', 'location.long']].to_numpy() for _, group in real]
        real_array = np.array(group_arrays)
        test_set = RawDataset(real_array)

        # Split the data into training and testing sets
        X_train, X_test = data, test_set
        train_loader = DataLoader(X_train, batch_size=16, num_workers=16)
        validation_loader = DataLoader(X_test, batch_size=8, num_workers=16)

        if not ABLATE_BOW:
            embed_epochs = wildgraph.EMBED_EPOCHS
            tokenizer = data.tokenizer
            embedder = wildgraph.train_embed(data.edge_index, epochs=embed_epochs)

            if ABLATE_NO_PE:
                model = wildgraph.train_wildgraph(data, embedder, tokenizer, train_loader, validation_loader, epochs, file_name=file_name, seq_length=self.seq_len, pe = 0)
            else:
                model = wildgraph.train_wildgraph(data, embedder, tokenizer, train_loader, validation_loader, epochs, file_name=file_name, seq_length=self.seq_len)
        else:
            embed_dim = len(tokenizer.classes_)
            def embed(x, embed_dim_func):
                embeddings = []
                for item in x:
                    x_embed = -1 * torch.ones(embed_dim)
                    if item >= 0:
                        x_embed[item] = 4
                    embeddings.append(x_embed)
                return torch.stack(embeddings)
            partial_embed = lambda x: embed(x, len(tokenizer.classes_))
            model = wildgraph.train_wildgraph(data, partial_embed, tokenizer, train_loader, validation_loader, epochs, file_name=file_name, seq_length=self.seq_len, embed_dim=len(tokenizer.classes_))
        #embed_dim = len(tokenizer.classes_)
        #return



        #model = wildgraph.train_wildgraph(data, embedder, tokenizer, train_loader, validation_loader, epochs, file_name=file_name, seq_length=self.seq_len, )
        model.eval()
        z_samples, z_init = model.process()
        generated = model.generate_traj(z_samples, z_init, self.test_sample_size)
        return generated

    def run_trasnformer(self, df_train, df_test, epochs):
        data = dataset.TrajDataset(df_train, len(train_index), seq_length=self.seq_len, is_tokenize_special_chars=True)
        tokenizer = data.tokenizer
        training_data = data.sentences
        model = transformer.Runner().run(tokenizer, training_data, epochs, self.seq_len)
        heat_map = heatmap.Heatmap(file_name)
        # Generate text
        max_tokens_to_generate = 65000
        #model = AutoregressiveWrapper.load_checkpoint('./trained_model_full_sentences.txt')
        generator = transformer.Generator(model, tokenizer)
        pad = tokenizer.transform(['<pad>'])[0]    
        generated_text = generator.generate(max_tokens_to_generate=max_tokens_to_generate, prompt="<sos>", padding_token=pad)

        generated = generated_text.replace('<pad> ', '').strip()
        sentences = generated.replace("<sos> ", "").strip()
        cells = sentences.split(".")
        cells = [s.strip() for s in cells]
        cells = [s.split(" ") for s in cells]
        cells = [sublist for sublist in cells if any(sublist)]
        filtered_cells = [[value for value in filter(lambda x: x != '', cell)] for cell in cells]
        #pdb.set_trace()
        padded_array =  [(sublist[:self.seq_len] if len(sublist) > self.seq_len else sublist + [sublist[-1]] * (self.seq_len - len(sublist))) for sublist in filtered_cells] # Make them seq len
        padded_array = padded_array[0:self.test_sample_size]

        #cells_traj = [tokenizer.inverse_transform(cell) for cell in filtered_cells]
        traj_long_lat = [[heat_map.sample(h3) for h3 in cell] for cell in padded_array]
        return traj_long_lat

    def run_vae(self, df_train, df_test, epochs):
        real = df_train[['label', 'location.lat', 'location.long']].groupby('label')
        group_arrays = [group[['location.lat', 'location.long']].to_numpy() for _, group in real]
        train_array = np.array(group_arrays)
        dataset = vae.CustomDataset(train_array)
        X_train = dataset

        train_loader = DataLoader(X_train, batch_size=8, num_workers=8)
        autoencoder = vae.LitAutoEncoder(self.seq_len)

        trainer = L.Trainer(max_epochs=epochs, devices=1)
        trainer.fit(autoencoder, train_loader)
        autoencoder.encoder.eval()

        x = torch.tensor(train_array)
        x = x.view(x.size(0), -1)
        # Fit GMM on encoded data
        encoded_data = autoencoder.encoder(x.float()).detach().cpu().numpy()
        autoencoder.gmm.fit(encoded_data)

        # Generate random samples from the GMM
        random_samples = autoencoder.gmm.sample(self.test_sample_size )[0]
        # Convert NumPy array to PyTorch tensor
        random_samples = torch.from_numpy(random_samples).float()

        # Decode the samples
        generated = autoencoder.decoder(random_samples).detach().cpu().numpy()
        generated = generated.reshape(self.test_sample_size , self.seq_len, 2)
        return generated

K_FOLD_COUNT = 5
TEST_SAMPLE_SIZE = 60

if __name__ == '__main__':
    L.seed_everything(2022)
    parser = ArgumentParser()
    parser.add_argument('--dataset', type=str, default='geese')
    parser.add_argument('--exp', type=str, default='WILDGRAPH') # GAN,VAE,WILDGEN,TRANSFORMER
    parser.add_argument('--epochs', type=int, default=90)
    parser.add_argument('--split_distance', type=float, default=.25) # maximum distance between 2 points in a given region (r)
    parser.add_argument('--num_exps', type=int, default=5)  # How many experiments to run
    parser.add_argument('--desc', type=str, default="full")  # An arbitrary name to describe the experiment
    parser.add_argument('--ablate', type=str, default="not-ablate")  # An arbitrary name to describe the experiment
    args = parser.parse_args()

    if (args.ablate == "uniform_coarse"):
        ABLATE_UNIFORM_COARSE = True
    elif (args.ablate == "uniform_fine"):
        ABLATE_UNIFORM_FINE = True
    elif (args.ablate == "no_pe"):
        ABLATE_NO_PE = True
    elif (args.ablate  == "bow"):
        ABLATE_BOW = True


    # Read File
    columns = ['idx', 'label', 'location.lat', 'location.long']
    file_name = f"data/{args.dataset}.csv"
    df = pd.read_csv(file_name, usecols=columns)
    
    # K-Fold split
    kf= KFold(n_splits=K_FOLD_COUNT)
    population_count = max(df['label'])+1
    X = range(population_count)

    # Initialize Metric
    hausdorff = dist.Hausdorff()
    dtw = dist.DTW()
    fde = dist.FDE()
    corr = dist.Corr()

    distances = [.1, .25, .5, .75, 1]
    if args.dataset == 'geese':
        likeness_clusters = 18
        seq_len = 185
    elif args.dataset == 'stork':
        likeness_clusters = 16
        seq_len = 79
    else:
        exit(-1)


    print("#########################################################")
    print("#########################################################")
    print("#########################################################")
    print(f"EXPERIMENT ({args.exp}): {args.dataset} EPOCHS={args.epochs }")
    print("#####")
    print("#####")
    print("#####")
    print("#####")
    runner = Runner(seq_len=seq_len, distance=args.split_distance, test_sample_size=TEST_SAMPLE_SIZE)

    logging.basicConfig(filename=f"./wild_experiments_log/{args.exp}/_{args.exp}_epochs={args.epochs}_{args.dataset}_{args.split_distance}_{args.desc}_{args.ablate}_{args.num_exps}-exps.log", level=logging.INFO)
    mean1_all, mean2_all, mean3_all, touched1_all, touched2_all, touched3_all, time_all, r_all, chi_all = 0, 0, 0, 0, 0, 0, 0, 0, 0
    r_full_all, chi_full_all = 0,0
    for rep in range(args.num_exps):
        time_exp = 0
        min1_c, max1_c, mean1_c, touched1_c = 0, 0, 0, 0 
        min2_c, max2_c, mean2_c, touched2_c = 0, 0, 0, 0 
        min3_c, max3_c, mean3_c, touched3_c = 0, 0, 0, 0 
        r_c, chi_c = 0, 0
        start_time = time.time()
        logging.info(f"###########{start_time} EXPERIMENT ({rep+1}): #################################### ")
        
        full = df[['label', 'location.lat', 'location.long']].groupby('label')
        full_set = [group[['location.lat', 'location.long']].to_numpy() for _, group in full]
        full_array = np.array(full_set)
        print("full array length = ", len(full_array))

        

        for k, (train_index, test_index) in enumerate(kf.split(X)):
            df_train = df[df['label'].isin(train_index)].copy()
            df_test = df[df['label'].isin(test_index)]
            test = df_test[['label', 'location.lat', 'location.long']].groupby('label')
            group_arrays = [group[['location.lat', 'location.long']].to_numpy() for _, group in test]
            test_array = np.array(group_arrays)
            print("test array length = ", len(test_array))
            #validation_loader = DataLoader(X_test, batch_size=16, num_workers=8)
            
            if (args.exp == "WILDGRAPH"):
                generated = runner.run_wildgraph(df_train, df_test, args.epochs)
            elif (args.exp == "TRANSFORMER"):
                generated = runner.run_trasnformer(df_train, df_test, args.epochs)
            elif (args.exp == "VAE"):
                generated = runner.run_vae(df_train, df_test, args.epochs)
            elif (args.exp == "GAN"):
                generated = gan.run_gan(df_train, args.epochs, seq_len)
            elif (args.exp == "WILDGEN"):
                mat_data = scipy.io.loadmat(f"data/wildgen/Traj_{args.dataset}_{k}.mat")
                data = np.reshape(mat_data[f"Traj_{args.dataset}_{k}"], (seq_len,2,1000))
                data = data[:,:,rep*200:(rep+1)*200]
                generated = wildgen.run_wildgen(df_train, data)
            elif (args.exp == "LEVY"):
                levy = pd.read_csv(f"./data/levy/levy-{args.dataset}-{k+1}.csv")
                levy = levy[['idx', 'lat', 'lng']].groupby('idx')
                gen_flat = [group[['lat', 'lng']].to_numpy() for _, group in levy]
                all = np.array(gen_flat)
                indexes = range(len(all))
                random = np.random.choice(indexes, size=60, replace=False)
                generated =  np.array([list(all[index]) for index in random])

            min1, max1, mean1, touched1 = hausdorff.update(test_array, generated)
            min2, max2, mean2, touched2 = dtw.update(test_array, generated)
            min3, max3, mean3, touched3 = fde.update(test_array, generated)
            r, chi = corr.update(test_array, generated, likeness_clusters)
            
            test_len = len(test_index)
            touched1, touched2, touched3 = touched1/ test_len, touched2/test_len, touched3/test_len

            #np.save(f"./wild_experiments_log/{args.exp}/{args.exp}_EPOCHS={args.epochs}_{rep}_k{k}_{args.dataset}.npy", generated)

            logging.info(f"k: {k},########### Means:  mean_haus: {mean1}, mean_dtw {mean2} mean_fde {mean3} ({touched1}, {touched2}, {touched3}) corr={r},{chi}")

            min1_c, max1_c, mean1_c, touched1_c = min1 + min1_c, max1 + max1_c, mean1_c + mean1, touched1_c + touched1
            min2_c, max2_c, mean2_c, touched2_c = min2 + min2_c, max2 + max2_c, mean2_c + mean2, touched2_c + touched2
            min3_c, max3_c, mean3_c, touched3_c = min3 + min3_c, max3 + max3_c, mean3_c + mean3, touched3_c + touched3
            r_c, chi_c = r_c+r, chi_c+chi
        r_full, chi_full = corr.update(full_array, generated, likeness_clusters)
        res = np.array([min1_c, max1_c, mean1_c, min2_c, max2_c, mean2_c, min3_c, max3_c, mean3_c, touched1_c, touched2_c, touched3_c, r_c, chi_c]) / (K_FOLD_COUNT)
        end_time = time.time()
        time_exp = end_time - start_time
        time_all = time_all + time_exp
        logging.info(f"{end_time} END Experiment {rep} FINAL: Means: (mean_haus, mean_dtw, mean_fde) {res[2]}, {res[5]}, {res[8]}, Time={time_exp}s {r_full} {chi_full}")
        logging.info({"  ":"  ", "k": "FINAL", "min_h": res[0], "max_h": res[1], "mean_haus": res[2], "min_d": res[3], "max_d": res[4], "mean_dtw": res[5], "min_f": res[6], "max_f": res[7], "mean_f": res[8]})
        mean1_all, mean2_all, mean3_all, touched1_all, touched2_all, touched3_all, time_all = mean1_all+res[2], mean2_all+res[5], mean3_all+res[8], touched1_all+res[9], touched2_all+res[10], touched3_all+res[11], time_all+time_exp
        r_all, chi_all = r_all +res[12], chi_all +res[13]
        r_full_all, chi_full_all = r_full + r_full_all, chi_full+ chi_full_all

    res = np.array([mean1_all, mean2_all, mean3_all, touched1_all, touched2_all, touched3_all, time_all, r_all, chi_all, r_full_all, chi_full_all]) / (args.num_exps)
    logging.info(f"REPORT STATS ALL EXP AVERAGES: Means: (mean_haus, mean_dtw, mean_fde) {res[0]}, {res[1]}, {res[2]}, ({res[3]}, {res[4]}, {res[5]}) full-corr={res[9]},{res[10]} corr={res[7]},{res[8]} Time={res[6]}s")
