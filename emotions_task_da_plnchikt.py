import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt 
import seaborn as sns

import re
from tqdm.notebook import tqdm
from datetime import datetime
from functools import reduce
from operator import add
from collections import Counter

import nltk
from nltk.tokenize import TreebankWordTokenizer, TweetTokenizer
nltk.download('stopwords')
from nltk.corpus import stopwords
stopwords = list(set(stopwords.words('english')))
from gensim.models.phrases import Phrases, Phraser
from string import punctuation
punctuation = list(punctuation)

!pip install pytorch_revgrad

import torch
from pytorch_revgrad import RevGrad

df_gossip = pd.read_csv('')
df_gossip

df_politi = pd.read_csv('')
df_politi

def ratio(x):
  if x == 'Joy':
    return 0
  elif x == 'Sadness':
    return 1
  elif x == 'Surprise':
    return 2
  elif x == 'Disgust':
    return 3
  elif x == 'Fear':
    return 4
  elif x == 'Trust':
    return 5
  elif x == 'Anticipation':
    return 6
  else:
    return 7

df_gossip['Emotion_label'] = df_gossip[''].apply(ratio)
df_gossip

df_politi['Emotion_label'] = df_politi[''].apply(ratio)
df_politi

stop = stopwords + punctuation + ['“','’', '“', '”', '‘','...']
tqdm.pandas()

def lowerizer(article):

  return article.lower()

def remove_html(article):

    article = re.sub("(<!--.*?-->)", "", article, flags=re.DOTALL)
    return article

def remove_url(article):

    article = re.sub(r'https?:\/\/.\S+', "", article)
    return article

def remove_hashtags(article):

    article = re.sub("#"," ",article)
    return article

def remove_a(article):

    article = re.sub("@"," ",article)
    return article

def remove_brackets(article):

    article = re.sub('\[[^]]*\]', '', article)
    return article

def remove_stop_punct(article):

    final_article = []
    for i in article.split():
        if i not in stop:
            final_article.append(i.strip())
    return " ".join(final_article)

def preprocessing(article):

    article = lowerizer(article)
    article = remove_html(article)
    article = remove_url(article)
    article = remove_hashtags(article)
    article = remove_a(article)
    article = remove_brackets(article)
    article = remove_stop_punct(article)
    return article

df_gossip['article_clean'] = df_gossip[''].progress_apply(lambda x : preprocessing(x))
df_gossip

df_politi['article_clean'] = df_politi[''].progress_apply(lambda x : preprocessing(x))
df_politi

import torch
from torch import nn
import torch.optim as optim
import nltk
from nltk import word_tokenize
nltk.download('punkt')
from torch.nn.utils.rnn import pad_sequence
import random
from torch.utils.data import Dataset, DataLoader

def get_splits(x, y, z, splits):

  n = len(x)
  indexes = np.arange(n)
  random.shuffle(indexes)
  valid_begin = int(splits[0]*n)
  test_begin = valid_begin + int(splits[1]*n)
  train_x, train_y, train_z = np.array(x)[indexes[:valid_begin]], np.array(y)[indexes[:valid_begin]], np.array(z)[indexes[:valid_begin]]
  valid_x, valid_y, valid_z = np.array(x)[indexes[valid_begin:test_begin]], np.array(y)[indexes[valid_begin:test_begin]], np.array(z)[indexes[valid_begin:test_begin]]
  test_x, test_y, test_z = np.array(x)[indexes[test_begin:]], np.array(y)[indexes[test_begin:]], np.array(z)[indexes[test_begin:]]
  return (train_x, train_y, train_z), (valid_x, valid_y, valid_z), (test_x, test_y, test_z)

splits = (0.8, 0.1)
(train_gossip_x, train_gossip_y, train_gossip_z), (valid_gossip_x, valid_gossip_y, valid_gossip_z), (test_gossip_x, test_gossip_y, test_gossip_z) = get_splits(df_gossip[''], df_gossip[''],df_gossip[''], splits)

splits = (0.8, 0.1)
(train_politi_x, train_politi_y, train_politi_z), (valid_politi_x, valid_politi_y, valid_politi_z), (test_politi_x, test_politi_y, test_politi_z) = get_splits(df_politi[''], df_politi[''], df_politi[''], splits)


class TextClassificationDataset(Dataset):
    def __init__(self, data, categories, emotion_catergories, vocab = None, max_length = 100, min_freq = 5):
        
        self.data = data
        self.max_length = max_length
        if vocab is not None:
            self.word2idx, self.idx2word = vocab
        else:
            self.word2idx, self.idx2word = self.build_vocab(self.data, min_freq)
      
        tokenized_data = [word_tokenize(file.lower()) for file in self.data]
        indexed_data = [[self.word2idx.get(word, self.word2idx['UNK']) for word in file] for file in tokenized_data]
        tensor_data = [torch.LongTensor(file) for file in indexed_data]
        tensor_y = torch.FloatTensor(categories)
        tensor_z = torch.LongTensor(emotion_catergories)
        cut_tensor_data = [tensor[:max_length] for tensor in tensor_data]
        self.tensor_data = pad_sequence(cut_tensor_data, batch_first=True, padding_value=0)
        self.tensor_y = tensor_y
        self.tensor_z = tensor_z
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        return self.tensor_data[idx], self.tensor_y[idx], self.tensor_z[idx] 
    
    def build_vocab(self, corpus, count_threshold):
        word_counts = {}
        for sent in corpus:
            for word in word_tokenize(sent.lower()):
                if word not in word_counts:
                    word_counts[word] = 0
                word_counts[word] += 1   
        filtered_word_counts = {word: count for word, count in word_counts.items() if count >= count_threshold}        
        words = sorted(filtered_word_counts.keys(), key=word_counts.get, reverse=True) + ['UNK']
        word_index = {words[i] : (i+1) for i in range(len(words))}
        idx_word = {(i+1) : words[i] for i in range(len(words))}
        return word_index, idx_word
    
    def get_vocab(self):
        return self.word2idx, self.idx2word

training_dataset_gossip = TextClassificationDataset(train_gossip_x, train_gossip_y, train_gossip_z)
training_word2idx_gossip, training_idx2word_gossip = training_dataset_gossip.get_vocab()
valid_dataset_gossip = TextClassificationDataset(valid_gossip_x, valid_gossip_y, valid_gossip_z, (training_word2idx_gossip, training_idx2word_gossip))
test_dataset_gossip = TextClassificationDataset(test_gossip_x, test_gossip_y, test_gossip_z ,(training_word2idx_gossip, training_idx2word_gossip))

training_dataloader_gossip = DataLoader(training_dataset_gossip, batch_size = 200, shuffle=True)
valid_dataloader_gossip = DataLoader(valid_dataset_gossip, batch_size = 25)
test_dataloader_gossip = DataLoader(test_dataset_gossip, batch_size = 25)

training_dataset_politi = TextClassificationDataset(train_politi_x, train_politi_y, train_politi_z)
training_word2idx_politi, training_idx2word_politi = training_dataset_politi.get_vocab()
valid_dataset_politi = TextClassificationDataset(valid_politi_x, valid_politi_y, valid_politi_z,(training_word2idx_politi, training_idx2word_politi))
test_dataset_politi = TextClassificationDataset(test_politi_x, test_politi_y, valid_politi_z, (training_word2idx_politi, training_idx2word_politi))

training_dataloader_politi = DataLoader(training_dataset_politi, batch_size = 200, shuffle=True)
valid_dataloader_politi = DataLoader(valid_dataset_politi, batch_size = 25)
test_dataloader_politi = DataLoader(test_dataset_politi, batch_size = 25)

import gensim.downloader as api
loaded_glove_model = api.load("glove-wiki-gigaword-300")
loaded_glove_embeddings = loaded_glove_model.vectors

def get_glove_adapted_embeddings(glove_model, input_voc):
  keys = {i: glove_model.vocab.get(w, None) for w, i in input_voc.items()}
  index_dict = {i: key.index for i, key in keys.items() if key is not None}
  embeddings = np.zeros((len(input_voc)+1,glove_model.vectors.shape[1]))
  for i, ind in index_dict.items():
      embeddings[i] = glove_model.vectors[ind]
  return embeddings

GloveEmbeddings_gossip = get_glove_adapted_embeddings(loaded_glove_model, training_word2idx_gossip)
GloveEmbeddings_politi = get_glove_adapted_embeddings(loaded_glove_model, training_word2idx_politi)

class LSTMModel(nn.Module):

    def __init__(self, embedding_dim, vocabulary_size, hidden_dim, embeddings=None, fine_tuning=False):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        if embeddings:
            self.embeddings_gossip = nn.Embedding.from_pretrained(torch.FloatTensor(GloveEmbeddings_gossip), freeze=not fine_tuning, padding_idx=0)
            self.embeddings_politi = nn.Embedding.from_pretrained(torch.FloatTensor(GloveEmbeddings_politi), freeze=not fine_tuning, padding_idx=0)
        else:
            self.embeddings = nn.Embedding(num_embeddings=vocabulary_size+1, embedding_dim=embedding_dim, padding_idx=0)

        self.lstm = nn.LSTM(input_size=embedding_dim, hidden_size=hidden_dim, batch_first=True, num_layers=2)
        self.linear_1 = nn.Linear(in_features=2*hidden_dim, out_features=1)
        self.grl = RevGrad()
        self.linear_2 = nn.Linear(in_features=2*hidden_dim, out_features=1)
        self.linear_emotion = nn.Linear(in_features=2*hidden_dim, out_features=8)

    def forward(self, inputs_a, inputs_b):
      emb_a = self.embeddings_gossip(inputs_a)
      emb_b = self.embeddings_politi(inputs_b)

      lstm_out_a, (ht_a, ct_a) = self.lstm(emb_a, None)
      h_a = torch.cat((ht_a[-2], ht_a[-1]), dim=1)
      x_a = torch.squeeze(self.linear_2(self.grl(h_a)))
      x_a_label = torch.squeeze(self.linear_1(h_a))
      x_a_emotion = torch.squeeze(self.linear_emotion(h_a))

      lstm_out_b, (ht_b, ct_b) = self.lstm(emb_b, None)
      h_b = torch.cat((ht_b[-2], ht_b[-1]), dim=1)
      x_b = torch.squeeze(self.linear_2(self.grl(h_b)))
      x_b_label = torch.squeeze(self.linear_1(h_b))
      x_b_emotion = torch.squeeze(self.linear_emotion(h_b))
      return x_a, x_b, x_a_label, x_b_label, x_a_emotion, x_b_emotion

def train_epoch(model, opt, criterion,criterion_emotions, dataloader_gossip, dataloader_politi):
  model.train()
  losses = []
  accs = []
  for i,  (item1, item2) in enumerate(zip(dataloader_gossip, dataloader_politi)):
      x_gossip, y_gossip, y_gossip_emotion = item1
      x_politi, y_politi, y_politi_emotion = item2 
      opt.zero_grad()
      pred_gossip, pred_politi, _ ,  pred_politi_label, pred_gossip_emotion, pred_politi_emotion = model(x_gossip, x_politi)
      loss_politi = criterion(pred_politi, torch.ones_like(y_politi))
      loss_gossip = criterion(pred_gossip, torch.zeros_like(y_gossip))
      loss_politi_label = criterion(pred_politi_label, y_politi)
      loss_politi_emotions = criterion_emotions(pred_politi_emotion, y_politi_emotion)
      loss_gossip_emotions = criterion_emotions(pred_gossip_emotion, y_gossip_emotion)
      alpha = 0.5
      loss_total = (loss_politi + loss_gossip) * alpha  + (loss_politi_label) * (1-alpha) + (alpha) * (loss_politi_emotions  + loss_gossip_emotions)
      loss_total.backward()
      opt.step()
      losses.append(loss_total.item())
      num_corrects = sum((torch.sigmoid(pred_politi_label)>0.5) == y_politi)
      acc = 100.0 * num_corrects/len(y_politi)
      accs.append(acc.item())
      if (i%20 == 0):
          print("Batch " + str(i) + " : training loss = " + str(loss_total.item()) + "; training acc = " + str(acc.item()))
  return losses, accs

def eval_model(model, criterion, evalloader_gossip, evalloader_politi):

  model.eval()
  total_epoch_loss = 0
  total_epoch_acc = 0
  preds = []

  total_epoch_loss_gossip = 0
  total_epoch_acc_gossip = 0
  preds_gossip = []
  with torch.no_grad():
      for i, (item1, item2) in enumerate(zip(evalloader_gossip, evalloader_politi)):
          x_gossip, y_gossip, _ = item1
          x_politi, y_politi, _ = item2
          _ , _ , pred_gossip_label ,  pred_politi_label, _, _ = model(x_gossip, x_politi)
          num_corrects = sum((torch.sigmoid(pred_politi_label)>0.5) == y_politi)
          acc = 100.0 * num_corrects/len(y_politi)
          total_epoch_acc += acc.item()
          preds.append(pred_politi_label)
          num_corrects = sum((torch.sigmoid(pred_gossip_label)>0.5) == y_gossip)
          acc = 100.0 * num_corrects/len(y_gossip)
          total_epoch_acc_gossip += acc.item()
          preds_gossip.append(pred_gossip_label)

  return total_epoch_acc/(i+1), preds, total_epoch_acc_gossip/(i+1), preds_gossip

def experiment(model, opt, criterion, criterion_emotions, num_epochs = 5):

  train_losses = []
  valid_losses = []
  train_accs = []
  valid_accs = []
  print("Beginning training...")
  for e in range(num_epochs):
      print("Epoch " + str(e+1) + ":")
      losses, accs = train_epoch(model, opt, criterion, criterion_emotions, training_dataloader_gossip, training_dataloader_politi)
      train_losses.append(losses)
      train_accs.append(accs)
      valid_acc, val_preds, valid_acc_gossip, val_preds_gossip = eval_model(model, criterion, valid_dataloader_gossip, valid_dataloader_politi)
      print("Epoch " + str(e+1) + " Politi Validation acc = " + str(valid_acc))
      print("Epoch " + str(e+1) + " Gossip Validation acc = " + str(valid_acc_gossip))
  test_acc_politi, test_preds_politi, test_acc_gossip, test_preds_gossip = eval_model(model,criterion , test_dataloader_gossip, test_dataloader_politi)
  print( "Politi Test Accuracy = " + str(test_acc_politi) + " ||  Gossip Test Accuracy: " + str(test_acc_gossip))
  return train_losses, valid_losses, test_acc_politi, train_accs, valid_accs, test_acc_gossip, test_preds_gossip

#Parameters
EMBEDDING_DIM = 300 
VOCAB_SIZE = len(training_word2idx_politi)
HIDDEN_DIM = 256 
learning_rate = 0.0025
num_epochs = 10

model_lstm = LSTMModel(EMBEDDING_DIM, VOCAB_SIZE, HIDDEN_DIM,  embeddings=True, fine_tuning=False)

opt = optim.Adam(model_lstm.parameters(), lr=learning_rate, betas=(0.9, 0.999))
criterion = nn.BCEWithLogitsLoss()
criterion_emotions = nn.CrossEntropyLoss()

model_lstm

train_losses_lstm, valid_losses_lstm, test_loss_lstm, train_accs_lstm, valid_accs_lstm, test_acc_lstm, test_preds_lstm = experiment(model_lstm, opt, criterion,criterion_emotions, num_epochs)