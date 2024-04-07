import random
from typing import List

import torch
import numpy as np

import matplotlib.pyplot as plt

import pdb
from util import dist, dataset, heatmap
from sklearn.model_selection import KFold
import pandas as pd
from torch.utils.data import Dataset, DataLoader
import logging


class TokenEmbedding(torch.nn.Module):
    """
    PyTorch module that converts tokens into embeddings.

    Input dimension is: (batch_size, sequence_length)
    Output dimension is: (batch_size, sequence_length, d_model)
    """

    def __init__(self, d_model, number_of_tokens):
        super().__init__()
        #pdb.set_trace()
        self.embedding_layer = torch.nn.Embedding(
            num_embeddings=number_of_tokens,
            embedding_dim=d_model
        )

    def forward(self, x):
        return self.embedding_layer(x)


class PositionalEncoding(torch.nn.Module):
    """
    Pytorch module that creates a positional encoding matrix. This matrix will later be added to the
    transformer's input embeddings to provide a sense of position of the sequence elements.
    """

    def __init__(self, d_model, max_sequence_length):
        super().__init__()
        self.d_model = d_model
        self.max_sequence_length = max_sequence_length
        self.positional_encoding = self.create_positional_encoding()

    def create_positional_encoding(self):
        """
        Creates a positional encoding matrix of size (max_sequence_length, d_model).
        """
        # Initialize positional encoding matrix
        positional_encoding = np.zeros((self.max_sequence_length, self.d_model))

        # Calculate positional encoding for each position and each dimension
        for pos in range(self.max_sequence_length):
            for i in range(0, self.d_model, 2):
                # Apply sin to even indices in the array; indices in Python start at 0 so i is even.
                positional_encoding[pos, i] = np.sin(pos / (10000 ** ((2 * i) / self.d_model)))

                if i + 1 < self.d_model:
                    # Apply cos to odd indices in the array; we add 1 to i because indices in Python start at 0.
                    positional_encoding[pos, i + 1] = np.cos(pos / (10000 ** ((2 * i) / self.d_model)))

        # Convert numpy array to PyTorch tensor and return it
        return torch.from_numpy(positional_encoding).float().to(get_device())

    def forward(self, x):
        """
        Adds the positional encoding to the input embeddings at the corresponding positions.
        """
        # Add positional encodings to input embeddings. The ":" indexing ensures we only add positional encodings up
        # to the length of the sequence in the batch. x.size(0) is the batch size, so this is a way to make sure
        # we're not adding extra positional encodings.
        return x + self.positional_encoding[:x.size(1), :]


class MaskedSelfAttention(torch.nn.Module):
    """
    Pytorch module for a self attention layer.
    This layer is used in the MultiHeadedSelfAttention module.

    Input dimension is: (batch_size, sequence_length, embedding_dimension)
    Output dimension is: (batch_size, sequence_length, head_dimension)
    """

    def __init__(self, embedding_dimension, head_dimension):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.head_dimension = head_dimension
        self.query_layer = torch.nn.Linear(embedding_dimension, self.head_dimension)
        self.key_layer = torch.nn.Linear(embedding_dimension, self.head_dimension)
        self.value_layer = torch.nn.Linear(embedding_dimension, self.head_dimension)
        self.softmax = torch.nn.Softmax(dim=-1)

    def forward(self, x, mask):
        """
        Compute the self attention.

        x dimension is: (batch_size, sequence_length, embedding_dimension)
        output dimension is: (batch_size, sequence_length, head_dimension)
        mask dimension is: (batch_size, sequence_length)

        mask values are: 0 or 1. 0 means the token is masked, 1 means the token is not masked.
        """

        # x dimensions are: (batch_size, sequence_length, embedding_dimension)
        # query, key, value dimensions are: (batch_size, sequence_length, head_dimension)
        query = self.query_layer(x)
        key = self.key_layer(x)
        value = self.value_layer(x)

        # Calculate the attention weights.
        # attention_weights dimensions are: (batch_size, sequence_length, sequence_length)
        attention_weights = torch.matmul(query, key.transpose(-2, -1))

        # Scale the attention weights.
        attention_weights = attention_weights / np.sqrt(self.head_dimension)

        # Apply the mask to the attention weights, by setting the masked tokens to a very low value.
        # This will make the softmax output 0 for these values.
        mask = mask.reshape(attention_weights.shape[0], 1, attention_weights.shape[2])
        attention_weights = attention_weights.masked_fill(mask == 0, -1e9)

        # Softmax makes sure all scores are between 0 and 1 and the sum of scores is 1.
        # attention_scores dimensions are: (batch_size, sequence_length, sequence_length)
        attention_scores = self.softmax(attention_weights)

        # The attention scores are multiplied by the value
        # Values of tokens with high attention score get highlighted because they are multiplied by a larger number,
        # and tokens with low attention score get drowned out because they are multiplied by a smaller number.
        # Output dimensions are: (batch_size, sequence_length, head_dimension)
        return torch.bmm(attention_scores, value)


class MaskedMultiHeadedSelfAttention(torch.nn.Module):
    """
    Pytorch module for a multi head attention layer.

    Input dimension is: (batch_size, sequence_length, embedding_dimension)
    Output dimension is: (batch_size, sequence_length, embedding_dimension)
    """

    def __init__(self, embedding_dimension, number_of_heads):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.head_dimension = embedding_dimension // number_of_heads
        self.number_of_heads = number_of_heads

        # Create the self attention modules
        self.self_attentions = torch.nn.ModuleList(
            [MaskedSelfAttention(embedding_dimension, self.head_dimension) for _ in range(number_of_heads)])

        # Create a linear layer to combine the outputs of the self attention modules
        self.output_layer = torch.nn.Linear(number_of_heads * self.head_dimension, embedding_dimension)

    def forward(self, x, mask):
        """
        Compute the multi head attention.

        x dimensions are: (batch_size, sequence_length, embedding_dimension)
        mask dimensions are: (batch_size, sequence_length)
        mask values are: 0 or 1. 0 means the token is masked, 1 means the token is not masked.
        """
        # Compute the self attention for each head
        # self_attention_outputs dimensions are:
        # (number_of_heads, batch_size, sequence_length, head_dimension)
        self_attention_outputs = [self_attention(x, mask) for self_attention in self.self_attentions]

        # Concatenate the self attention outputs
        # self_attention_outputs_concatenated dimensions are:
        # (batch_size, sequence_length, number_of_heads * head_dimension)
        concatenated_self_attention_outputs = torch.cat(self_attention_outputs, dim=2)

        # Apply the output layer to the concatenated self attention outputs
        # output dimensions are: (batch_size, sequence_length, embedding_dimension)
        return self.output_layer(concatenated_self_attention_outputs)


class FeedForward(torch.nn.Module):
    """
    Pytorch module for a feed forward layer.

    A feed forward layer is a fully connected layer with a ReLU activation function in between.
    """

    def __init__(self, embedding_dimension, feed_forward_dimension):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.feed_forward_dimension = feed_forward_dimension
        self.linear_1 = torch.nn.Linear(embedding_dimension, feed_forward_dimension)
        self.linear_2 = torch.nn.Linear(feed_forward_dimension, embedding_dimension)

    def forward(self, x):
        """
        Compute the feed forward layer.
        """
        return self.linear_2(torch.relu(self.linear_1(x)))


class DecoderLayer(torch.nn.Module):
    """
    Pytorch module for an encoder layer.

    An encoder layer consists of a multi-headed self attention layer, a feed forward layer and dropout.

    Input dimension is: (batch_size, sequence_length, embedding_dimension)
    Output dimension is: (batch_size, sequence_length, embedding_dimension)
    """

    def __init__(
            self,
            embedding_dimension,
            number_of_heads,
            feed_forward_dimension,
            dropout_rate
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_heads = number_of_heads
        self.feed_forward_dimension = feed_forward_dimension
        self.dropout_rate = dropout_rate

        self.multi_headed_self_attention = MaskedMultiHeadedSelfAttention(embedding_dimension, number_of_heads)
        self.feed_forward = FeedForward(embedding_dimension, feed_forward_dimension)
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.layer_normalization_1 = torch.nn.LayerNorm(embedding_dimension)
        self.layer_normalization_2 = torch.nn.LayerNorm(embedding_dimension)

    def forward(self, x, mask):
        """
        Compute the encoder layer.

        x dimensions are: (batch_size, sequence_length, embedding_dimension)
        mask dimensions are: (batch_size, sequence_length)
        mask values are: 0 or 1. 0 means the token is masked, 1 means the token is not masked.
        """

        # Layer normalization 1
        normalized_x = self.layer_normalization_1(x)

        # Multi headed self attention
        attention_output = self.multi_headed_self_attention(normalized_x, mask)

        # Residual output
        residual_output = x + attention_output

        # Layer normalization 2
        normalized_residual_output = self.layer_normalization_2(residual_output)

        # Feed forward
        feed_forward_output = self.feed_forward(normalized_residual_output)

        # Dropout
        if self.training:
            feed_forward_output = self.dropout(feed_forward_output)

        # Residual output
        return residual_output + feed_forward_output


class DecoderStack(torch.nn.Module):
    """
    The decoder stack consists of multiple decoder layers in sequence.
    """

    def __init__(
            self,
            embedding_dimension,
            number_of_layers,
            number_of_heads,
            feed_forward_dimension,
            dropout_rate,
            max_sequence_length
    ):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_layers = number_of_layers
        self.number_of_heads = number_of_heads
        self.feed_forward_dimension = feed_forward_dimension
        self.dropout_rate = dropout_rate
        self.max_sequence_length = max_sequence_length

        # Create the encoder layers
        self.encoder_layers = torch.nn.ModuleList(
            [DecoderLayer(embedding_dimension, number_of_heads, feed_forward_dimension, dropout_rate) for _ in
             range(number_of_layers)])

    def forward(self, x, mask):
        decoder_outputs = x
        for decoder_layer in self.encoder_layers:
            decoder_outputs = decoder_layer(decoder_outputs, mask)

        return decoder_outputs


class LMHead(torch.nn.Module):
    """
    Pytorch module for the language model head.
    The language model head is a linear layer that maps the embedding dimension to the vocabulary size.
    """

    def __init__(self, embedding_dimension, number_of_tokens):
        super().__init__()
        self.embedding_dimension = embedding_dimension
        self.number_of_tokens = number_of_tokens
        self.linear = torch.nn.Linear(embedding_dimension, number_of_tokens)

    def forward(self, x):
        """
        Compute the language model head.

        x dimensions are: (batch_size, sequence_length, embedding_dimension)
        output dimensions are: (batch_size, sequence_length, number_of_tokens)
        """
        # Compute the linear layer
        # linear_output dimensions are: (batch_size, sequence_length, number_of_tokens)
        linear_output = self.linear(x)

        return linear_output


class LanguageModel(torch.nn.Module):
    """
    Pytorch module for a language model.
    """
    def __init__(
            self,
            number_of_tokens,  # The number of tokens in the vocabulary
            max_sequence_length=512,  # The maximum sequence length to use for attention
            embedding_dimension=512,  # The dimension of the token embeddings
            number_of_layers=6,  # The number of decoder layers to use
            number_of_heads=4,  # The number of attention heads to use
            feed_forward_dimension=None,  # The dimension of the feed forward layer
            dropout_rate=0.1  # The dropout rate to use
    ):
        super().__init__()
        self.number_of_tokens = number_of_tokens
        self.max_sequence_length = max_sequence_length
        self.embedding_dimension = embedding_dimension
        self.number_of_layers = number_of_layers
        self.number_of_heads = number_of_heads

        if feed_forward_dimension is None:
            # GPT-2 paper uses 4 * embedding_dimension for the feed forward dimension
            self.feed_forward_dimension = embedding_dimension * 4
        else:
            self.feed_forward_dimension = feed_forward_dimension

        self.dropout_rate = dropout_rate

        # Create the token embedding layer
        self.token_embedding = TokenEmbedding(embedding_dimension, number_of_tokens)

        # Create the positional encoding layer
        self.positional_encoding = PositionalEncoding(embedding_dimension, max_sequence_length)

        # Create the normalization layer
        self.layer_normalization = torch.nn.LayerNorm(embedding_dimension)

        # Create the decoder stack
        self.decoder = DecoderStack(
            embedding_dimension=embedding_dimension,
            number_of_layers=number_of_layers,
            number_of_heads=number_of_heads,
            feed_forward_dimension=self.feed_forward_dimension,
            dropout_rate=dropout_rate,
            max_sequence_length=max_sequence_length
        )

        # Create the language model head
        self.lm_head = LMHead(embedding_dimension, number_of_tokens)

    def forward(self, x, mask):
        # Compute the token embeddings
        # token_embeddings dimensions are: (batch_size, sequence_length, embedding_dimension)
        token_embeddings = self.token_embedding(x)

        # Compute the positional encoding
        # positional_encoding dimensions are: (batch_size, sequence_length, embedding_dimension)
        positional_encoding = self.positional_encoding(token_embeddings)

        # Post embedding layer normalization
        positional_encoding_normalized = self.layer_normalization(positional_encoding)

        decoder_outputs = self.decoder(positional_encoding_normalized, mask)
        lm_head_outputs = self.lm_head(decoder_outputs)

        return lm_head_outputs

    def save_checkpoint(self, path):
        print(f'Saving checkpoint {path}')
        torch.save({
            'number_of_tokens': self.number_of_tokens,
            'max_sequence_length': self.max_sequence_length,
            'embedding_dimension': self.embedding_dimension,
            'number_of_layers': self.number_of_layers,
            'number_of_heads': self.number_of_heads,
            'feed_forward_dimension': self.feed_forward_dimension,
            'dropout_rate': self.dropout_rate,
            'model_state_dict': self.state_dict()
        }, path)

    @staticmethod
    def load_checkpoint(path) -> 'LanguageModel':
        checkpoint = torch.load(path)
        model = LanguageModel(
            number_of_tokens=checkpoint['number_of_tokens'],
            max_sequence_length=checkpoint['max_sequence_length'],
            embedding_dimension=checkpoint['embedding_dimension'],
            number_of_layers=checkpoint['number_of_layers'],
            number_of_heads=checkpoint['number_of_heads'],
            feed_forward_dimension=checkpoint['feed_forward_dimension'],
            dropout_rate=checkpoint['dropout_rate']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        return model.to(get_device())


class AutoregressiveWrapper(torch.nn.Module):
    """
    Pytorch module that wraps a GPT model and makes it autoregressive.
    """

    def __init__(self, gpt_model):
        super().__init__()
        self.model = gpt_model
        self.max_sequence_length = self.model.max_sequence_length

    def forward(self, x, mask):
        """
        Autoregressive forward pass
        """
        inp, target = x[:, :-1], x[:, 1:]
        mask = mask[:, :-1]

        output = self.model(inp, mask)
        return output, target

    def next_token_probabilities(self, x, mask, temperature=1.0):
        """
        Calculate the token probabilities for the next token in the sequence.
        """
        logits = self.model(x, mask)[:, -1]
        #pdb.set_trace()
        # Apply temperature
        if temperature != 1.0:
            logits = logits / temperature

        # Apply the softmax
        probabilities = torch.softmax(logits, dim=-1)

        return probabilities

    def save_checkpoint(self, path):
        self.model.save_checkpoint(path)

    @staticmethod
    def load_checkpoint(path) -> 'AutoregressiveWrapper':
        model = LanguageModel.load_checkpoint(path)
        return AutoregressiveWrapper(model).to(get_device())


def get_device():
    return torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class Trainer:

    def __init__(self, model, tokenizer, optimizer=None):
        super().__init__()
        self.model = model
        if optimizer is None:
            self.optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        else:
            self.optimizer = optimizer
        self.tokenizer = tokenizer
        self.loss_function = torch.nn.CrossEntropyLoss()

    def train(self, data: List[str], epochs, batch_size):
        loss_per_epoch = []
        for epoch in range(epochs):
            losses = []

            # Shuffle the sequences
            random.shuffle(data)

            # Create batches of sequences and their respective mask.
            batches = []
            for i in range(0, len(data), batch_size):
                sequence_tensor = torch.tensor(data[i: i + batch_size], dtype=torch.long)

                # Create the mask tensor for the batch, where 1 means the token is not a padding token
                mask_tensor = torch.ones_like(sequence_tensor)
                pad = self.tokenizer.transform(['<pad>'])[0]
                mask_tensor[sequence_tensor == pad] = 0

                batches.append((sequence_tensor, mask_tensor))

            # Train the model on each batch
            for batch in batches:
                self.model.train()

                # Create the input and mask tensors
                input_tensor = torch.zeros((batch_size, self.model.max_sequence_length + 1), dtype=torch.long)
                mask_tensor = torch.zeros((batch_size, self.model.max_sequence_length + 1), dtype=torch.long)

                for i, input_entry in enumerate(batch[0]):
                    input_tensor[i] = input_entry

                for i, mask_entry in enumerate(batch[1]):
                    mask_tensor[i] = mask_entry

                # Compute the model output
                model_output, target = self.model.forward(
                    x=input_tensor.to(get_device()),
                    mask=mask_tensor.to(get_device())
                )

                # Compute the losses
                # The loss is computed on the model output and the target
                loss = self.loss_function(model_output.transpose(1, 2), target)
                # loss = self.loss_function(model_output[:, -1, :], target[:, -1])

                # Backpropagate the loss.
                loss.backward()

                # Clip the gradients. This is used to prevent exploding gradients.
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)

                # Update the model parameters. This is done by taking a step in the direction of the gradient.
                self.optimizer.step()

                # Reset the gradients. This is done so that the gradients from the previous batch
                # are not used in the next step.
                self.optimizer.zero_grad()

                # Append the loss to the list of losses, so that the average loss can be computed for this epoch.
                losses.append(loss.item())

            # Print the loss
            epoch_loss = np.average(losses)
            loss_per_epoch.append(epoch_loss)
            print('Epoch:', epoch, 'Loss:', epoch_loss)

        return loss_per_epoch


class Generator:

    def __init__(self, model, tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    def generate( self, max_tokens_to_generate: int, prompt: str = None, temperature: float = 1.0, eos_token: int = None, padding_token: int = 0):
        self.model.eval()
        if prompt is None:
            start_tokens = self.tokenizer.transform(["<sos>"])
        else:
            start_tokens = self.tokenizer.transform([prompt])
        #pdb.set_trace()
        input_tensor = torch.tensor(
            pad_left( sequence=start_tokens, final_length=self.model.max_sequence_length + 1, padding_token=padding_token), dtype=torch.long
        ).to(get_device())

        num_dims = len(input_tensor.shape)

        if num_dims == 1:
            input_tensor = input_tensor[None, :]

        out = input_tensor
        for _ in range(max_tokens_to_generate):
            x = out[:, -self.model.max_sequence_length:]
            mask = torch.ones_like(x)
            mask[x == padding_token] = 0
            #pdb.set_trace()
            # Compute the next token probabilities
            next_token_probabilities = self.model.next_token_probabilities(
                x=x,
                temperature=temperature,
                mask=mask
            )
            #pdb.set_trace()
            # Sample the next token from the probability distribution
            next_token = torch.multinomial(next_token_probabilities, num_samples=1)

            # Append the next token to the output
            out = torch.cat([out, next_token], dim=1)

            # If the end of sequence token is reached, stop generating tokens
            if eos_token is not None and next_token == eos_token:
                break

        generated_tokens = out[0].tolist()
        #pdb.set_trace()
        return ' '.join(self.tokenizer.inverse_transform(generated_tokens) )


def create_training_sequences(max_sequence_length, tokenized_training_data):
    # Create sequences of length max_sequence_length + 1
    # The last token of each sequence is the target token
    sequences = []
    for i in range(0, len(tokenized_training_data) - max_sequence_length - 1):
        sequences.append(tokenized_training_data[i: i + max_sequence_length + 1])
    return sequences


def tokenize_and_pad_training_data(max_sequence_length, tokenizer, training_data):
    # Tokenize the training data
    #pdb.set_trace()
    #tokenized_training_data = tokenizer.tokenize(training_data)
    pad = tokenizer.transform(['<pad>'])[0]
    for _ in range(max_sequence_length):
        # Prepend padding tokens
        training_data.insert(0, pad)
    return training_data


class Runner(torch.nn.Module):

    def __init__(self):
        super().__init__()

    def run(self, tokenizer, training_data, epochs, seq_length):
        embedding_dimension = 256
        max_sequence_length = seq_length

        #pdb.set_trace()
        number_of_tokens = len(tokenizer.classes_) + 1
        print("number of tokens", number_of_tokens)

        # Create the model
        model = AutoregressiveWrapper(LanguageModel(
            embedding_dimension=embedding_dimension,
            number_of_tokens=number_of_tokens,
            number_of_heads=4,
            number_of_layers=3,
            dropout_rate=0.1,
            max_sequence_length=max_sequence_length
        )).to(get_device())


        tokenized_and_padded_training_data = tokenize_and_pad_training_data(max_sequence_length, tokenizer, training_data)
        sequences = create_training_sequences(max_sequence_length, tokenized_and_padded_training_data)

        # Train the model
        optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
        trainer = Trainer(model, tokenizer, optimizer)
        loss_per_epoch = trainer.train(sequences, epochs=epochs, batch_size=16)

        # Plot the loss per epoch in log scale
        plt.plot(loss_per_epoch)
        plt.yscale('log')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.show()

        return model
        #model.save_checkpoint('./trained_model_full_sentences.txt')
        #pdb.set_trace()

def pad_left(sequence, final_length, padding_token):
    #pdb.set_trace()
    return [padding_token] * (final_length - len(sequence)) + list(sequence)

"""
#tokenizer = Tokenizer()
#model = AutoregressiveWrapper.load_checkpoint('./trained_model_full')
hausdorff = dist.Hausdorff()
dtw = dist.DTW()
fde = dist.FDE()
heatmap = heatmap.Heatmap('trajs.csv')
class RawDataset(Dataset):
    def __init__(self, df):
        self.trajs = df
    def __len__(self):
        return len(self.trajs)
    def __getitem__(self, index):
        return self.trajs[index]
    
file_path = 'trajs.csv'
columns = ['idx', 'label', 'location.lat', 'location.long']
df = pd.read_csv(file_path, usecols=columns)

fold_count = 5
kf= KFold(n_splits=fold_count)
X = range(60)
SAMPLES_TEST = 60
EPOCHS = 50

logging.basicConfig(filename=f"wildbm_transformer_{EPOCHS}.log", level=logging.INFO)

for rep in range(5):
    min1_c, max1_c, mean1_c = 0, 0, 0 
    min2_c, max2_c, mean2_c = 0, 0, 0 
    min3_c, max3_c, mean3_c = 0, 0, 0 
    logging.info(f"EXPERIMENT ${rep}: ####################################")
    for k, (train_index, test_index) in enumerate(kf.split(X)):
        df_train = df[df['label'].isin(train_index)].copy()
        data = dataset.TrajDataset(df_train, len(train_index), seq_length=185, is_tokenize_special_chars=True)
        tokenizer = data.tokenizer
        training_data = data.sentences

        df_test = df[df['label'].isin(test_index)]
        real = df[['label', 'location.lat', 'location.long']].groupby('label')
        group_arrays = [group[['location.lat', 'location.long']].to_numpy() for _, group in real]
        real_array = np.array(group_arrays)
        test_set = RawDataset(real_array)

        model = Runner().run(tokenizer, training_data, EPOCHS)


        # Generate text
        max_tokens_to_generate = 65000
        #model = AutoregressiveWrapper.load_checkpoint('./trained_model_full_sentences.txt')
        generator = Generator(model, tokenizer)
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
        padded_array =  [(sublist[:185] if len(sublist) > 185 else sublist + [sublist[-1]] * (185 - len(sublist))) for sublist in filtered_cells] # Make them 185
        padded_array = padded_array[0:60]

        #cells_traj = [tokenizer.inverse_transform(cell) for cell in filtered_cells]
        traj_long_lat = [[heatmap.sample(h3) for h3 in cell] for cell in padded_array]

        min1, max1, mean1 = hausdorff.update(traj_long_lat, real_array)
        min2, max2, mean2 = dtw.update(traj_long_lat, real_array)
        min3, max3, mean3 = fde.update(traj_long_lat, real_array)

        logging.info({"Means:": "",  "mean_haus": mean1, "mean_dtw": mean2, "mean_fde": mean3, "gen_len": len(traj_long_lat)})
        #logging.info({"k": k, "min_h": min1, "max_h": max1, "mean_haus": mean1, "min_d": min2, "max_d": max2, "mean_dtw": mean2, "min_f": min3, "max_f": max3, "mean_f": mean3})
        #logging.info("K1 ", min1, max1, mean1, min2, max2, mean2, min3, max3, mean3)
        min1_c, max1_c, mean1_c = min1 + min1_c, max1 + max1_c, mean1_c + mean1
        min2_c, max2_c, mean2_c = min2 + min2_c, max2 + max2_c, mean2_c + mean2
        min3_c, max3_c, mean3_c = min3 + min3_c, max3 + max3_c, mean3_c + mean3
    
    res = np.array([min1_c, max1_c, mean1_c, min2_c, max2_c, mean2_c, min3_c, max3_c, mean3_c]) / (fold_count)
    logging.info(f"END Experiment {rep} FINAL: Means: (mean_haus, mean_dtw, mean_fde) {res[2]}, {res[5]}, {res[8]}")
    logging.info({"  ":"  ", "k": "FINAL", "min_h": res[0], "max_h": res[1], "mean_haus": res[2], "min_d": res[3], "max_d": res[4], "mean_dtw": res[5], "min_f": res[6], "max_f": res[7], "mean_f": res[8]})
"""