#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jun 29 19:02:34 2024

@author: mayrabonacelly
"""
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score, roc_curve
import matplotlib.pyplot as plt
import json
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from sklearn.preprocessing import OneHotEncoder
import numpy as np
import torch.nn.functional as F
from sklearn.ensemble import VotingClassifier
from sklearn.utils import resample
import math
from PIL import Image, ImageDraw, ImageFont
from sklearn.model_selection import StratifiedKFold
from sklearn.utils import resample
import torchbnn as bnn 
import torch.distributions.kl as kl
from sklearn.metrics import classification_report
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
from collections import Counter, defaultdict
from torchvision import models
from sklearn.cluster import KMeans
import faiss 
from sklearn.cluster import KMeans
from torch.distributions.kl import kl_divergence  
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import iqr, spearmanr
import torch, numpy as np, random
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


vocab_size = 5001  
pad_idx = 5000 
lambda_l1=0.01 #lambda of L1 regularization


def visualise_data(file=None, df=None):
    if df is None:
        if file is None:
            raise ValueError("Either a file or a DataFrame must be provided")
        data = []
        with open(file, 'r') as file:
            for line in file:
                data.append(json.loads(line))
        df = pd.DataFrame(data)

    df['text_length'] = df['text'].apply(len)


    
def truncate_texts(fl, column='text_length'):
    
    data = []
    with open(fl, 'r') as file:
      for line in file:
           data.append(json.loads(line))
    df = pd.DataFrame(data)

    df['text_length'] = df['text'].apply(len)
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1

    upper_whisker = Q3 + 1.5 * IQR

    def truncate_text(text):
        max_length = int(upper_whisker)
        return text[:max_length] if len(text) > max_length else text

    df['text'] = df['text'].apply(truncate_text)
    df['text_length'] = df['text'].apply(len) 

    return df    
       
    
def inject_noise(texts, target_size, noise_level=0.1):
    augmented_texts = []
    current_size = len(texts)
    while len(augmented_texts) < (target_size - current_size):
        for text in texts:
            if len(augmented_texts) >= (target_size - current_size):
                break
            new_text = text.copy()  
            num_changes = int(len(new_text) * noise_level) if len(new_text) > 0 else 0
            for _ in range(num_changes):
                idx_to_modify = np.random.randint(len(new_text))
                new_text[idx_to_modify] = np.random.randint(1, 5001) 
            augmented_texts.append(new_text)
    return [{'text': text} for text in texts + augmented_texts] 


def split_data_d1(df): 
    if 'model' in df.columns:
        df = df.drop('model', axis=1)
    df['text_length'] = df['text'].apply(len)
    average_length =  int(df['text_length'].mean())
    longest_length =  df['text_length'].max()
    shortest_length =  df['text_length'].min()
    ratio = int(len(df[df['label']==0])) / len(df[df['label']==1])
    train, global_test = train_test_split(df, test_size = 0.1, random_state=42, stratify=df['label']) #stratify=labels_df1
    val, test = train_test_split(global_test, test_size = 0.95, random_state=42, stratify=global_test['label'])
    return train, val, test, ratio 
 
def split_data_d2_augmented(df):
    if 'model' in df.columns:
        df = df.drop('model', axis=1)

    df_ai = df[df['label'] == 0]
    df_human = df[df['label'] == 1]

    test_size_per_class = min(len(df_ai), len(df_human)) // 10  
    test_ai = df_ai.sample(n=test_size_per_class, random_state=42)
    test_human = df_human.sample(n=test_size_per_class, random_state=42)
    test = pd.concat([test_ai, test_human])
    
    df_train_val = df.drop(test.index)

    num_human_samples_needed = int(len(df_train_val[df_train_val['label'] == 1]) * 3)  # Increase by 2.1x
    bootstrapped_human_samples = resample(df_human, replace=True, n_samples=num_human_samples_needed, random_state=42)

    noise_level = 0.02  # Reduce noise to keep samples realistic
    augmented_human_samples = inject_noise(bootstrapped_human_samples['text'].tolist(), target_size=num_human_samples_needed, noise_level=noise_level)

    df_human_augmented = pd.DataFrame(augmented_human_samples)
    df_human_augmented['label'] = 1  

    num_ai_samples_to_keep = int(len(df_train_val[df_train_val['label'] == 0]) * 0.70)
    df_ai_train_val = df_train_val[df_train_val['label'] == 0].sample(n=num_ai_samples_to_keep, random_state=42)
    train_val_combined = pd.concat([df_ai_train_val, df_human_augmented])
    train, val = train_test_split(train_val_combined, test_size=0.2, random_state=42, stratify=train_val_combined['label'])

    ratio = round(len(train[train['label'] == 0]) / len(train[train['label'] == 1]), 2)
    return train, val, test, ratio

def split_data_d2_augmented2(df):
    if 'model' in df.columns:
        df = df.drop('model', axis=1)

    df_ai = df[df['label'] == 0]
    df_human = df[df['label'] == 1]

    test_size_per_class = min(len(df_ai), len(df_human)) // 10  

    test_ai = df_ai.sample(n=test_size_per_class, random_state=42)
    test_human = df_human.sample(n=test_size_per_class, random_state=42)
    test = pd.concat([test_ai, test_human])

    df_train_val = df.drop(test.index)

    num_ai_samples_to_keep = int(len(df_train_val[df_train_val['label'] == 0]) * 0.70)  # 70% of AI samples
    num_human_samples_needed = int(len(df_train_val[df_train_val['label'] == 1]) * 2.1)  # Increase human samples 2.1

    df_ai_train_val = df_train_val[df_train_val['label'] == 0].sample(n=num_ai_samples_to_keep, random_state=42)
    augmented_human_samples = inject_noise(df_human['text'].tolist(), target_size=num_human_samples_needed)

    df_human_augmented = pd.DataFrame(augmented_human_samples)
    df_human_augmented['label'] = 1  # Assign human label

    train_val_combined = pd.concat([df_ai_train_val, df_human_augmented])

    train, val = train_test_split(train_val_combined, test_size=0.2, random_state=42, stratify=train_val_combined['label'])

    ratio = round(len(train[train['label'] == 0])/len(train[train['label'] == 1]),2)
     
    #print("Training set size: ", len(train), " Validation set size: ", len(val), " Test set size: ", len(test))
    #print("Training ai samples: ", len(train[train['label'] == 0]), " training human samples: ", len(train[train['label'] == 1]), " ratio: ", int(len(train[train['label'] == 0])/len(train[train['label'] == 1])))
    #print("Validation ai samples: ", len(val[val['label'] == 0]), " validation human samples: ", len(val[val['label'] == 1]), " ratio: ", int(len(val[val['label'] == 0])/len(val[val['label'] == 1])))
    #print("Test ai samples: ", len(test[test['label'] == 0]), " test human samples: ", len(test[test['label'] == 1]), " ratio: ", int(len(test[test['label'] == 0])/len(test[test['label'] == 1])))

    return train, val, test, ratio

def monte_carlo_bayes_fc(bayes_fc, input_tensor, num_samples=5):
    """
    Perform Monte Carlo sampling through the Bayesian Linear Layer.
    Returns mean and variance of predictions.
    """
    outputs = []
    for _ in range(num_samples):
        outputs.append(bayes_fc(input_tensor).unsqueeze(0))  # Shape [1, batch_size, 1]

    outputs = torch.cat(outputs, dim=0)  # Shape [num_samples, batch_size, 1]
    mean = outputs.mean(dim=0).squeeze(1)       # [batch_size]
    variance = outputs.var(dim=0).squeeze(1)    # [batch_size]

    return mean, variance


class TextDataset(Dataset):
    def __init__(self, dataframe):
        self.dataframe = dataframe
        self.lenghts = []
        self.instance_ids = [f"instance_{i}" for i in range(len(dataframe))]  

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        text = self.dataframe.iloc[index]['text']
        label = self.dataframe.iloc[index]['label']
        instance_id = self.instance_ids[index]  


        if not isinstance(text, list):
            text = [text]
        if not text:
            text = [0]
        text_tensor = torch.tensor(text, dtype=torch.long)
        label_tensor = torch.tensor(label, dtype=torch.float)


        return text_tensor, label_tensor, instance_id 


    def collate_batch(self, batch):
        batch.sort(key=lambda x: len(x[0]), reverse=True)
        sequences, labels, instance_ids = zip(*batch)  


        lenghts = [len(seq) for seq in sequences]
        sequences = [torch.tensor(seq) if not isinstance(seq, torch.Tensor) else seq for seq in sequences]
        sequences_padded = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=5000)


        self.lenghts = lenghts
        return sequences_padded, torch.tensor(labels), lenghts, list(instance_ids)  
       
def generateDataLoader(train_data, val_data, test_data, batch_size=50):
      train_dataset = TextDataset(train_data)
      train_dataloader = DataLoader(train_dataset,
                                  batch_size=batch_size,
                                  shuffle=True,
                                  collate_fn=train_dataset.collate_batch)


      val_dataset = TextDataset(val_data)
      val_dataloader = DataLoader(val_dataset,
                                batch_size=batch_size,
                                shuffle=False,
                                collate_fn=val_dataset.collate_batch)


      test_dataset = TextDataset(test_data)
      test_dataloader = DataLoader(test_dataset,
                                 batch_size=batch_size,
                                 shuffle=False,
                                 collate_fn=test_dataset.collate_batch)  


      return train_dataloader, val_dataloader, test_dataloader

class GlobalAttention(nn.Module):
    def __init__(self, hidden_dim):
        super(GlobalAttention, self).__init__()
        self.hidden_dim = hidden_dim
        self.attn = nn.Linear(2 * hidden_dim, hidden_dim) 
        self.v = nn.Parameter(torch.rand(hidden_dim)) 

    def forward(self, hidden, encoder_outputs, mask):
        batch_size = hidden.size(0)
        src_len = encoder_outputs.size(1)
        hidden_exp = hidden.unsqueeze(1).expand(-1, src_len, -1) 
        energy = torch.tanh(self.attn(torch.cat((hidden_exp, encoder_outputs), dim=2))) 
        v = self.v.unsqueeze(0).unsqueeze(0).expand(encoder_outputs.size(0),-1,-1) 
        energy=energy.transpose(1,2) 
        attention_scores = torch.bmm(v, energy) 
        attention_scores = attention_scores.squeeze(1) 
        attn_w = attention_scores.masked_fill(mask == 0, -1e10) 
        attn_w = F.softmax(attn_w, dim=1) 
        return attn_w


class DeepCluster(nn.Module):
    def __init__(self, embedding_dim, num_clusters=10):
        super(DeepCluster, self).__init__()
        
        self.num_clusters = num_clusters

        self.feature_extractor = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),  # Final projected space (learned instead of PCA)
            nn.ReLU()
        )

        self.cluster_head = nn.Linear(32, num_clusters, bias=False)

        self.kmeans = None

    def forward(self, x):
        features = self.feature_extractor(x)  
        cluster_scores = self.cluster_head(features)
        return features, cluster_scores

    def fit_predict(self, inputs):
        features, _ = self.forward(inputs)
        features_np = features.detach().cpu().numpy()

        features_np = features_np / (np.linalg.norm(features_np, axis=1, keepdims=True) + 1e-6)

        try:
            self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
            cluster_labels = self.kmeans.fit_predict(features_np)
        except Exception as e:
            print(f" Clustering failed: {e}")
            cluster_labels = np.zeros(len(features_np), dtype=int)

        return cluster_labels

    def predict(self, query):
        if self.kmeans is None:
            print(" KMeans not trained yet. Returning default cluster -1.")
            return torch.tensor(-1)

        query_np = query.detach().cpu().numpy().reshape(1, -1)
        query_np = query_np / (np.linalg.norm(query_np, axis=1, keepdims=True) + 1e-6)

        try:
            cluster_id = self.kmeans.predict(query_np)[0]
        except Exception as e:
            print(f"Cluster prediction failed: {e}")
            cluster_id = -1

        return torch.tensor(cluster_id)

    def update_clusters(self, new_memories):
        """Recalculates KMeans centroids based on new memory features."""
        new_features, _ = self.forward(new_memories)
        new_features_np = new_features.detach().cpu().numpy()
        new_features_np = new_features_np / (np.linalg.norm(new_features_np, axis=1, keepdims=True) + 1e-6)

        if self.kmeans is None:
            print("No previous KMeans model. Initializing fresh clustering.")
            self.kmeans = KMeans(n_clusters=self.num_clusters, random_state=42, n_init=10)
        else:
            print(" Updating KMeans clustering with new features.")

        self.kmeans.fit(new_features_np)


class HybridAlphaHead(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        
        self.qm_layer = nn.Linear(input_dim * 2, 16)
        self.cos_layer = nn.Linear(1, 16)
        self.combined_layer = nn.Linear(32, 2)
        
        self.cos_weight = nn.Parameter(torch.ones(1))

    def forward(self, query, memory, cosine_sim):
        qm = F.relu(self.qm_layer(torch.cat([query, memory], dim=1)))
        cos = self.cos_layer(cosine_sim)
        combined = torch.cat([qm, cos], dim=1)
        
        alpha_params = self.combined_layer(combined)
        alpha_logits = alpha_params[:, 0]
        
        cosine_sim_scaled = torch.sigmoid(cosine_sim)
        
        alpha_logits += self.cos_weight * cosine_sim_scaled.squeeze()
        
        alpha_mean = torch.sigmoid(alpha_logits)
        alpha_mean = torch.clamp(alpha_mean, 1e-3, 1 - 1e-3)
        
        alpha_concentration = F.softplus(alpha_params[:, 1]) + 1.0
        
        return alpha_mean, alpha_concentration, alpha_params


class DynamicMemoryBank:
    def __init__(self, memory_dim, num_clusters=10, min_cluster_size=5,
                 memory_bank_size=8000, cluster_threshold=800):
        self.memory_dim = memory_dim
        self.num_clusters = num_clusters
        self.min_cluster_size = min_cluster_size
        self.memory_bank_size = memory_bank_size
        self.cluster_threshold = cluster_threshold

        self.memory_dict = {}  
        self.cluster_labels = {}  
        self.retrieval_counts = {i: 0 for i in range(num_clusters)}
        self.new_memory_count = 0  

        self.kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
        self.deep_cluster = DeepCluster(embedding_dim=memory_dim, num_clusters=num_clusters)

        self.needs_fitting = False
        self.last_cluster_update_epoch = 0
        self.current_epoch = None

        self.cluster_centroids = {}

    def store(self, instance_id, new_memory):
        if not isinstance(new_memory, torch.Tensor):
            new_memory = torch.tensor(new_memory, dtype=torch.float32)

        self.memory_dict[instance_id] = new_memory.detach()
        self.new_memory_count += 1

        if self.new_memory_count >= self.cluster_threshold:
            self.needs_fitting = True

        if len(self.memory_dict) > self.memory_bank_size:
            excess_count = len(self.memory_dict) - self.memory_bank_size
            sorted_keys = sorted(self.memory_dict.keys())
            for key in sorted_keys[:excess_count]:
                del self.memory_dict[key]


    def retrieve(self, instance_id, query, is_training=True, current_epoch=None, seq_len=None):
        retrieval_source = None

        if is_training and self.needs_fitting:
            self.fit_clusters()

        if len(self.memory_dict) == 0:
            return query, torch.tensor(0.5, device=query.device), "cold_fallback"

        debug_info = {}

        if instance_id in self.memory_dict:
            retrieved_memory = self.memory_dict[instance_id]
            cluster = self.cluster_labels.get(instance_id, -1)
            if cluster in self.retrieval_counts:
                self.retrieval_counts[cluster] += 1
            debug_info["retrieval_source"] = "direct_match"
            retrieval_source = "direct_match"
        else:
            if self.cluster_labels:
                random_instance = np.random.choice(list(self.cluster_labels.keys()))
                cluster_id = self.cluster_labels[random_instance]
            else:
                cluster_id = -1

            cluster_instances = [
                k for k, v in self.cluster_labels.items()
                if v == cluster_id and k in self.memory_dict
            ]

            if cluster_instances:
                retrieved_memory = self.memory_dict[
                    np.random.choice(cluster_instances)
                ]
                if cluster_id in self.retrieval_counts:
                    self.retrieval_counts[cluster_id] += 1
                debug_info["retrieval_source"] = "cluster_fallback"
                retrieval_source = "cluster_fallback"
            else:
                retrieved_memory = query
                debug_info["retrieval_source"] = "cold_fallback"
                retrieval_source = "cold_fallback"

        eps = 1e-8
        query_norm = query.norm(p=2).item()
        memory_norm = retrieved_memory.norm(p=2).item()

        if query_norm < eps or memory_norm < eps:
            cos_sim = 0.0
            debug_info["cosine_warning"] = "zero_norm_detected"
        else:
            cos_sim = F.cosine_similarity(
                query.unsqueeze(0), retrieved_memory.unsqueeze(0)
            ).item()

        reliability = torch.sigmoid(torch.tensor(cos_sim)).to(query.device)
        reliability = torch.clamp(reliability, min=0.01, max=0.99)

        debug_info.update({
            "query_norm": round(query_norm, 4),
            "memory_norm": round(memory_norm, 4),
            "cos_sim": round(cos_sim, 4),
            "reliability": round(reliability.item(), 4),
        })
        
        cluster_id = self.cluster_labels.get(instance_id, -1)
        centroid_distance = None
        cluster_var_distance = None
        if cluster_id >= 0 and cluster_id in self.cluster_centroids:
           centroid = self.cluster_centroids[cluster_id]
           centroid_distance = torch.norm(query.detach().cpu() - centroid.detach().cpu()).item()
           if hasattr(self, "intra_cluster_metrics") and cluster_id in self.intra_cluster_metrics:
              cluster_var_distance = self.intra_cluster_metrics[cluster_id]["var_distance"]
        if not hasattr(self, "memory_retrieval_debug_log"):
            self.memory_retrieval_debug_log = []

        self.memory_retrieval_debug_log.append({
            "epoch": current_epoch,
            "instance_id": instance_id,
            "seq_len": seq_len,  # add sequence length
            "retrieval_source": debug_info.get("retrieval_source"),
            "cosine_sim": debug_info.get("cos_sim"),
            "reliability": debug_info.get("reliability"),
            "query_norm": debug_info.get("query_norm"),
            "memory_norm": debug_info.get("memory_norm"),
            "cluster_id": self.cluster_labels.get(instance_id, -1),
            "centroid_distance": centroid_distance,
            "cluster_var_distance": cluster_var_distance
        })
        return retrieved_memory, reliability, retrieval_source

    def fit_clusters(self):
        if not self.needs_fitting or len(self.memory_dict) < self.min_cluster_size:
            return

        instance_ids = list(self.memory_dict.keys())
        memories_np = np.array(
            [self.memory_dict[k].cpu().numpy() for k in instance_ids]
        )
        memories_tensor = torch.tensor(memories_np, dtype=torch.float32)

        cluster_labels = self.deep_cluster.fit_predict(memories_tensor)
        if cluster_labels is None or len(cluster_labels) == 0:
            cluster_labels = np.zeros(len(instance_ids), dtype=int)

        self.cluster_labels = {
            instance_id: label
            for instance_id, label in zip(instance_ids, cluster_labels)
        }
        self.needs_fitting = False
        self.last_cluster_update_epoch += 1
        
        self.compute_cluster_centroids()
        self.intra_cluster_metrics = self.compute_intra_cluster_metrics()


    def update_memory(self):

        if len(self.memory_dict) == 0:
            return

        if self.new_memory_count >= self.cluster_threshold:
            self.fit_clusters()

        if len(self.memory_dict) > self.memory_bank_size:
            excess_count = len(self.memory_dict) - self.memory_bank_size
            sorted_keys = sorted(self.memory_dict.keys())
            for key in sorted_keys[:excess_count]:
                del self.memory_dict[key]

    def compute_cluster_centroids(self):
        from collections import defaultdict

        cluster_vectors = defaultdict(list)

        for instance_id, memory_vec in self.memory_dict.items():
            cluster_label = self.cluster_labels.get(instance_id, -1)
            if cluster_label >= 0:
                cluster_vectors[cluster_label].append(memory_vec)

        cluster_centroids = {}
        for cluster_id, vectors in cluster_vectors.items():
            stacked = torch.stack(vectors)
            centroid = stacked.mean(dim=0)
            cluster_centroids[cluster_id] = centroid

        self.cluster_centroids = cluster_centroids

 

        return cluster_centroids
  
    def find_nearest_examples_to_centroids(self, top_k=5):

     results = {}
     for cluster_id, centroid in self.cluster_centroids.items():
        neighbors = []
        for instance_id, vec in self.memory_dict.items():
            if self.cluster_labels.get(instance_id, -1) == cluster_id:
                sim = F.cosine_similarity(
                    centroid.unsqueeze(0),
                    vec.unsqueeze(0)
                ).item()
                neighbors.append((instance_id, sim, vec))
        
        neighbors = sorted(neighbors, key=lambda x: -x[1])
        top_neighbors = neighbors[:top_k]
        results[cluster_id] = top_neighbors

        print(f"Cluster {cluster_id} top examples:")
        for instance_id, sim, _ in top_neighbors:
            print(f"  - ID {instance_id} | Cosine sim {sim:.4f}")

     return results
 
    def compute_intra_cluster_metrics(self):

     results = {}

     for cluster_id, centroid in self.cluster_centroids.items():
        members = [
            vec for inst_id, vec in self.memory_dict.items()
            if self.cluster_labels.get(inst_id, -1) == cluster_id
        ]

        if not members:
            continue

        member_tensors = torch.stack(members)

        diffs = member_tensors - centroid
        dists = torch.norm(diffs, dim=1)
        mean_dist = dists.mean().item()
        var_dist = dists.var().item()

        cos_sims = F.cosine_similarity(
            member_tensors,
            centroid.unsqueeze(0),
            dim=1
        )
        mean_cos_sim = cos_sims.mean().item()

        results[cluster_id] = {
            "mean_distance": mean_dist,
            "var_distance": var_dist,
            "mean_cosine_similarity": mean_cos_sim,
            "count": len(members)
        }

        #print(f"Cluster {cluster_id}:")
        #print(f"  - Members: {len(members)}")
        #print(f"  - Mean distance to centroid: {mean_dist:.4f}")
        #print(f"  - Variance of distances: {var_dist:.4f}")
        #print(f"  - Mean cosine similarity: {mean_cos_sim:.4f}")
        #print()

     return results


class BayesIntuit(nn.Module):
   #LSTMWithAttention
   def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers=1,
                bidirectional=False, dropout=0, memory_bank_size=50,
                num_clusters=8, projected_dim=None, use_alpha_in_output = True, use_memory=True, use_clipped_alpha_in_prediction=True, trust_memory_percent = 0.2):
       super(BayesIntuit, self).__init__()

       self.use_memory = use_memory
       
       # === Perception Stage ===
       self.embedding = nn.Embedding(vocab_size, embedding_dim)
       self.lstm = nn.LSTM(
           embedding_dim, hidden_dim, num_layers=num_layers,
           batch_first=True, bidirectional=bidirectional, dropout=dropout
       )
       # === Attention Mechanism ===
       lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
       self.attention_layer = GlobalAttention(lstm_output_dim)

       self.fc_attention = nn.Linear(lstm_output_dim, lstm_output_dim)
       # === Projection Layer
       self.use_projection = projected_dim is not None
       self.projector = nn.Linear(lstm_output_dim, projected_dim) if self.use_projection else None
       memory_dim = projected_dim if self.use_projection else lstm_output_dim
       # === Memory Bank
       self.memory_bank = DynamicMemoryBank(memory_dim=memory_dim, num_clusters=num_clusters)
       # === Bayesian Decision Stage
       self.bayes_fc = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                                       in_features=memory_dim, out_features=1)

       # === Alpha Estimation (Hybrid Head with Cosine Similarity)
       self.alpha_head = HybridAlphaHead(input_dim=memory_dim)
       self.context_evaluation = None  # No longer needed separately
       self.alpha_values = []
       self.cosine_similarities = []
       self.reliabilities = []  
       self.use_alpha_in_output = use_alpha_in_output 
       
       self.use_memory = use_memory
       if self.use_memory == False and self.use_alpha_in_output == True:
          self.use_memory = True
       self.use_clipped_alpha_in_prediction = use_clipped_alpha_in_prediction
       self.trust_memory_percent = trust_memory_percent 
       
       # === Stronger Prior (More Confidence around alpha = 0.2) ===
       # Mean = 10 / (10 + 40) = 0.2
       # Variance ≈ 0.0032 (narrower)
       self.alpha_1 = nn.Parameter(torch.tensor(10.0), requires_grad=True)
       self.alpha_2 = nn.Parameter(torch.tensor(40.0), requires_grad=True)
       self.debug_memory_retrieval_logs = []
       self.max_debug_logs = 25  #

  
   def apply_projection(self, tensor):
      if self.use_projection and self.projector is not None:
               return self.projector(tensor)
      return tensor

   def forward(self, text, text_lengths, mask, instance_ids=None,
           is_training=True, current_epoch=None, max_epochs=None, freeze_alpha=False):    
       embedded = self.embedding(text) #batch size, seq len, emb dim
       packed_embedded = pack_padded_sequence(embedded, text_lengths, batch_first=True)
       packed_output, (hidden, cell) = self.lstm(packed_embedded)
       output, _ = pad_packed_sequence(packed_output, batch_first=True)
       dense_outputs = self.bayes_fc(output)
       
       if self.lstm.bidirectional: hidden = torch.cat((hidden[-2,:,:], hidden[-1,:,:]), dim=1)
       else: hidden = hidden[-1,:,:] 
       attn_weights = self.attention_layer(hidden, output, mask).unsqueeze(1)

       attn_output = torch.bmm(attn_weights, output).squeeze(1) 
       attn_output_processed = F.relu(self.fc_attention(attn_output)) 
       attn_output_processed  = F.dropout(attn_output_processed, p=0.2, training= self.training)  
       attn_output_processed = self.apply_projection(attn_output_processed)
       
       # === Memory Retrieval ===
       if self.use_memory:
          retrieved_memories, reliabilities, retrieval_sources = [], [], []
    
          for i in range(attn_output_processed.size(0)):
             instance_id = instance_ids[i]
             seq_len = text_lengths[i]
             result = self.memory_bank.retrieve(instance_id, attn_output_processed[i], is_training=is_training, current_epoch=current_epoch, seq_len=seq_len)

             if isinstance(result, tuple):
                 if len(result) == 3:
                    retrieved_memory, reliability, retrieval_source = result
                 else:
                    retrieved_memory, reliability = result
                    retrieval_source = "unknown"
             else:
                    retrieved_memory = result
                    reliability = torch.tensor(0.5)
                    retrieval_source = "unknown"
            
             retrieved_memory = self.apply_projection(retrieved_memory)
             retrieved_memories.append(retrieved_memory)
             reliabilities.append(reliability)
             retrieval_sources.append(retrieval_source)
    
          retrieved_memory = torch.stack(retrieved_memories)
    
          reliabilities = torch.stack([
             r.clone().detach().float().to(attn_output_processed.device)
             if isinstance(r, torch.Tensor)
             else torch.tensor(r, dtype=torch.float32, device=attn_output_processed.device)
             for r in reliabilities
          ]).view(-1, 1)
          
       else:
         retrieved_memory = torch.zeros_like(attn_output_processed)
         reliabilities = torch.ones_like(attn_output_processed[:, :1]) * 0.5
         retrieval_sources = ["no_memory_used"] * attn_output_processed.size(0)
       
       # === Alpha Computation
       cosine_sim_raw = F.cosine_similarity(attn_output_processed, retrieved_memory, dim=1).view(-1, 1)
       # Standardized cosine similarity (used in HybridAlphaHead)
       cosine_sim = (cosine_sim_raw - cosine_sim_raw.mean()) / (cosine_sim_raw.std() + 1e-8)
       
       #Pure alpha (Informativeness)
       cosine_sim_scaled = cosine_sim_raw  
       cosine_sim_scaled = torch.sigmoid(4.0 * (cosine_sim_raw - 0.5))
       alpha_pure_mean = cosine_sim_scaled.view(-1, 1)
       alpha_pure_concentration = (1.0 + 20.0 * cosine_sim_scaled).view(-1, 1)
       
       alpha1_pure = None
       alpha2_pure = None

       #Alpha for prediction
       self.cosine_similarities.extend(cosine_sim.detach().cpu().flatten().tolist())

       alpha_mean, alpha_concentration, _ = self.alpha_head(attn_output_processed, retrieved_memory, cosine_sim)
       alpha_mean = torch.clamp(alpha_mean, 1e-3, 1 - 1e-3)
       alpha_concentration = torch.clamp(alpha_concentration, min=1.0)
       
       
       if freeze_alpha:
            alpha = alpha_mean.view(-1, 1)  # Use learned deterministic mean
            self.alpha_1 = None
            self.alpha_2 = None
       else:
            alpha_1 = alpha_mean * alpha_concentration
            alpha_2 = (1 - alpha_mean) * alpha_concentration
            alpha_dist = torch.distributions.Beta(alpha_1, alpha_2)
            alpha = alpha_dist.rsample().view(-1, 1)
            alpha = torch.clamp(alpha, 0.0, 1.0)
            alpha_beta = alpha
            self.alpha_1_buffer = alpha_1.detach()
            self.alpha_2_buffer = alpha_2.detach()
            
        #Informative Alpha
       alpha1_pure = alpha_pure_mean * alpha_pure_concentration
       alpha2_pure = (1 - alpha_pure_mean) * alpha_pure_concentration
       alpha_pure_dist = torch.distributions.Beta(alpha1_pure, alpha2_pure)
       alpha_informative = alpha_pure_dist.rsample()
       alpha_informative = torch.clamp(alpha_informative, 0.0, 1.0)
       alpha_beta_informative = alpha_informative

       # === Track values safely
       if hasattr(self, "alpha_values"):
          self.alpha_values.extend(alpha_mean.detach().cpu().tolist())
       if hasattr(self, "reliability_values"):
          self.reliability_values.extend(reliabilities.detach().cpu().numpy().flatten().tolist())
       if hasattr(self, "alpha_samples_all"):
          self.alpha_samples_all.extend(alpha.detach().cpu().flatten().tolist())
       if hasattr(self, "alpha_informative_samples_all"):
          self.alpha_informative_samples_all.extend(alpha_informative.detach().cpu().flatten().tolist())
        
       # === Blending Alpha with Reliability
       alpha = alpha * reliabilities + (1 - reliabilities) * 0.5
       alpha = torch.clamp(alpha, 0.01, 0.99)
       
       #Informative Alpha (optional)
       alpha_informative = alpha_informative * reliabilities + (1 - reliabilities) * 0.5
       alpha_informative = torch.clamp(alpha_informative, 0.01, 0.99)
       
       if not self.use_alpha_in_output:
          alpha = torch.zeros_like(alpha)

        
       if self.use_memory:
          alpha_clipped = torch.clamp(alpha, min=0.01, max=0.99)

          for i in range(attn_output_processed.size(0)):
             instance_id = instance_ids[i]
             memory_i = retrieved_memory[i]
             attn_i = attn_output_processed[i]
             if memory_i.shape != attn_i.shape:
               print(f"[ERROR] Shape mismatch in memory update — memory_i: {memory_i.shape}, attn_i: {attn_i.shape}")
               continue
             alpha_i = alpha_clipped[i]
             updated_memory = attn_i + (memory_i * alpha_i)
             self.memory_bank.store(instance_id, updated_memory)
        
        
       alpha_clipped = torch.clamp(alpha, min=0.01, max=0.99)
       alpha_clipped_informative = torch.clamp(alpha_informative, min=0.01, max=0.99)
       combined_output = attn_output_processed + (alpha_clipped * retrieved_memory) # Additively
       #combined_output = (attn_output_processed * (1-alpha_clipped)) + (alpha_clipped * retrieved_memory) #Convex combinatory
       dense_outputs = self.bayes_fc(combined_output) #Applying RelLU to the combined vector of attention output and model vector to avoid vanishing gradient 

       if not hasattr(self, "debug_memory_retrieval_logs"):
          self.debug_memory_retrieval_logs = []
       if len(self.debug_memory_retrieval_logs) < 2000 and is_training:
          log_entry = {
            "epoch": current_epoch,
             "instance_id": instance_ids[i] if instance_ids is not None else None,
             "cosine_sim_raw": cosine_sim_raw[i].item(),
             "cosine_sim": cosine_sim[i].item(),
              "alpha_mean": alpha_mean[i].item(),
              "alpha_concentration": alpha_concentration[i].item(),
              "alpha_pure_mean": alpha_pure_mean[i].item(),
              "alpha_pure_concentration": alpha_pure_concentration[i].item(),
              "alpha_1": alpha_1[i].item(),
              "alpha_2": alpha_2[i].item(),
              "alpha1_pure": alpha1_pure[i].item(),
              "alpha2_pure": alpha2_pure[i].item(),
              "alpha_beta": alpha_beta[i].item(),
              "alpha_beta_informative": alpha_beta_informative[i].item(),
              "reliability": reliabilities[i].item(),
              "alpha": alpha_clipped[i].item(),
              "alpha_informative": alpha_clipped_informative[i].item(),
             "attn_output_sample": attn_output_processed[i].detach().cpu().numpy().tolist()[:5],  
             "retrieved_memory_sample": retrieved_memory[i].detach().cpu().numpy().tolist()[:5],
             "retrieval_source": retrieval_sources[i]
          }
          self.debug_memory_retrieval_logs.append(log_entry)
       #dense_outputs = self.bayes_fc(attn_output_processed) #Applying RelLU to the combined vector of attention output and model vector to avoid vanishing gradient 
       alpha_entropy = - (alpha * torch.log(alpha + 1e-8) + (1 - alpha) * torch.log(1 - alpha + 1e-8)).mean()
       extra_debug_info = {
           "cosine_sim_raw": cosine_sim_raw.detach(),
           "cosine_sim": cosine_sim.detach(),
           "alpha_mean": alpha_mean.detach(),
           "alpha_informative_mean": alpha_pure_mean.detach()
           }

       if is_training:
          return dense_outputs, alpha_entropy, extra_debug_info
       else:
          return dense_outputs, None , extra_debug_info

def bayesian_train(traindata, valdata, model, criterion=torch.nn.BCEWithLogitsLoss(), lr=0.01, num_epochs=3, WithAttention=True, threshold=0.5, patience=2):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []
    
    alpha_means_epoch, alpha_vars_epoch = [], []
    reliability_means_epoch, reliability_vars_epoch = [], []
    alpha1_means_epoch, alpha2_means_epoch = [], []
    
    best_val_loss = float('inf')
    best_val_auc = 0
    epochs_no_improve = 0
    early_stop = False
    
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        correct_predictions = 0
        total_predictions = 0
        
        model.alpha_values = []
        model.cosine_similarities = []
        model.reliability_values = []
        model.debug_alpha_params = []

        for inputs_train, labels_train, lengths_train, instance_ids in traindata:
            optimizer.zero_grad()
            mask_train = (inputs_train != pad_idx) if WithAttention else None
            
            if hasattr(model, "memory_bank"):
               model.memory_bank.current_epoch = epoch


            outputs, alpha_entropy, debug_info = model(inputs_train, lengths_train, mask_train,
                                           instance_ids=instance_ids, is_training=True,
                                           current_epoch=epoch, max_epochs=num_epochs)
            outputs = outputs.squeeze(1)
            # Bayesian loss function (ELBO)
            likelihood_loss = criterion(outputs, labels_train.float())

            # === KL Divergence for Bayesian Layer
            kl_loss = kl.kl_divergence(
                torch.distributions.Normal(model.bayes_fc.weight_mu, torch.exp(0.5*model.bayes_fc.weight_log_sigma)),
                torch.distributions.Normal(model.bayes_fc.prior_mu, model.bayes_fc.prior_sigma)  # Add the prior distribution
            ).sum()

            
            # === KL Divergence for Alpha (if present)
            if hasattr(model, 'alpha_1') and hasattr(model, 'alpha_2') and model.alpha_1 is not None:
               prior_alpha = torch.distributions.Beta(
                             torch.tensor(2.0, device=model.alpha_1.device),
                             torch.tensor(2.0, device=model.alpha_1.device) #8.0
               )
               alpha_dist = torch.distributions.Beta(model.alpha_1, model.alpha_2)
               kl_alpha = torch.distributions.kl_divergence(alpha_dist, prior_alpha).mean()
            else:
               kl_alpha = 0.0     
               
            cosine_sim_raw = debug_info["cosine_sim_raw"]
            cosine_sim = debug_info["cosine_sim"]
            alpha_mean = debug_info["alpha_mean"]
            alpha_informative_mean = debug_info["alpha_informative_mean"]
            alpha_mean = torch.clamp(alpha_mean, 1e-3, 1 - 1e-3)
            alpha_informative_mean = torch.clamp(alpha_informative_mean, 1e-3, 1 - 1e-3)


            loss = likelihood_loss + 0.01 * kl_loss  # Weighted KL term
            kl_alpha_weight = 0.0  #0.001
            alpha_entropy_weight = 0.0 #0.001
            loss += kl_alpha_weight * kl_alpha
            loss += alpha_entropy_weight * alpha_entropy
            
            info_loss = F.mse_loss(alpha_mean.view(-1), alpha_informative_mean.view(-1))
            info_loss_weight = 0.05 #0.05
            loss += info_loss_weight * info_loss
            
            total_loss += loss.item()
            loss.backward()
            optimizer.step()

            predicted_labels = (torch.sigmoid(outputs) > 0.5).float()
            correct_predictions += (predicted_labels == labels_train).sum().item()
            total_predictions += labels_train.size(0)

        average_loss = total_loss / len(traindata)
        accuracy = correct_predictions / total_predictions
        train_accuracies.append(accuracy)
        train_losses.append(average_loss)
        
        model.eval()
        val_loss = 0
        val_correct_predictions = 0
        val_total_predictions = 0
        all_labels = []
        all_predictions = []
        with torch.no_grad():
            for inputs_val, labels_val, lengths_val, instance_ids in valdata:
                mask_val = (inputs_val != pad_idx) if WithAttention else None
                outputs_val, _ , debug_info = model(inputs_val, lengths_val, mask_val,
                                       instance_ids=instance_ids, is_training=False,
                                       current_epoch=epoch, max_epochs=num_epochs)
                outputs_val = outputs_val.squeeze(1)
                likelihood_val_loss = criterion(outputs_val, labels_val.float())
                
                kl_loss_val = kl.kl_divergence(
                    torch.distributions.Normal(model.bayes_fc.weight_mu, torch.exp(0.5*model.bayes_fc.weight_log_sigma)),
                    torch.distributions.Normal(model.bayes_fc.prior_mu, model.bayes_fc.prior_sigma)  # Adding here the prior distribution
                ).sum()
                
                loss = likelihood_val_loss + 0.00 * kl_loss_val  # Weighted KL term...
                val_loss += loss.item()
                

                predicted_labels_val = (torch.sigmoid(outputs_val) > threshold).float()
                val_correct_predictions += (predicted_labels_val == labels_val).sum().item()
                val_total_predictions += labels_val.size(0)
                
                all_labels.extend(labels_val.tolist())
                all_predictions.extend(predicted_labels_val.tolist())

        val_avg_loss = val_loss / len(valdata)
        val_accuracy = val_correct_predictions / val_total_predictions
        val_accuracies.append(val_accuracy)
        val_losses.append(val_avg_loss)
        
        roc_auc_val = roc_auc_score(all_labels, [p > 0.5 for p in all_predictions])
        
        
        alpha_array = np.array(model.alpha_values) if model.alpha_values else np.zeros(total_predictions)
        alpha1_array = model.alpha_1.detach().cpu().numpy() if model.alpha_1 is not None else np.zeros_like(alpha_array)
        alpha2_array = model.alpha_2.detach().cpu().numpy() if model.alpha_2 is not None else np.zeros_like(alpha_array)
        reliability_array = np.array(model.reliability_values) if model.reliability_values else np.full_like(alpha_array, fill_value=0.5)

        alpha_means_epoch.append(alpha_array.mean())
        alpha_vars_epoch.append(alpha_array.var())
        alpha1_means_epoch.append(alpha1_array.mean())
        alpha2_means_epoch.append(alpha2_array.mean())
        reliability_means_epoch.append(reliability_array.mean())
        reliability_vars_epoch.append(reliability_array.var())
        
        print(f"[Epoch {epoch+1}]")
        print(f"  Losses -> Likelihood: {likelihood_loss.item():.4f}")
        #print(f"  Alpha1: {model.alpha_1.mean().item():.4f}, Alpha2: {model.alpha_2.mean().item():.4f}")
        print(f"  Reliability Mean: {np.mean(model.reliability_values):.4f}")
        print(f"Alpha Mean (Epoch {epoch+1}): {alpha_array.mean():.4f}, Alpha Var: {alpha_array.var():.4f}, "
              f"Reliability Mean: {reliability_array.mean():.4f}, Alpha1: {alpha1_array.mean():.4f}, Alpha2: {alpha2_array.mean():.4f}")

        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        print(f'Epoch {epoch+1}: Training Accuracy: {accuracy:.4f}%, Training Loss: {average_loss:.4f}%, Validation Accuracy: {val_accuracy:.4f}%, Validation Loss: {val_avg_loss:.4f}%, Validation ROC-AUC: {roc_auc_val:.4f}%')
        if epochs_no_improve == patience:
            print("Early stopping triggered.")
            early_stop = True
            break
    
    if hasattr(model, "debug_memory_retrieval_logs"):
      with open("memory_debug_logs.json", "w") as f:
        json.dump(model.debug_memory_retrieval_logs, f, indent=2)

      N = 10000
      logs_df = pd.DataFrame(model.debug_memory_retrieval_logs[:N])
      logs_df.to_csv("memory_retrieval_debug_LSTM.csv", index=False)
      print(logs_df)
      
    if hasattr(model, "memory_bank") and hasattr(model.memory_bank, "memory_retrieval_debug_log"):
       df_log = pd.DataFrame(model.memory_bank.memory_retrieval_debug_log)
       df_log.to_csv("memory_retrieval_traceLSTM.csv", index=False)
       print("Memory retrieval log saved to memory_retrieval_trace.csv")

    return model, train_accuracies, val_accuracies, train_losses, val_losses


def test_performance_bayesian_with_uncertainty(testdata, model, criterion, threshold=0.5, n_samples=20):
    print("Testing IntuiBayes with Uncertainty Estimation...")
    model.eval()

    all_mean_preds = []
    all_std_preds = []
    all_labels = []
    all_probabilities = []
    instance_uncertainty = {}

    with torch.no_grad():
        for inputs_test, labels_test, lengths_test, instance_ids in testdata:
            mask_test = (inputs_test != pad_idx)
            preds_samples = []

            for _ in range(n_samples):
                outputs_test, _ , debug_info = model(inputs_test, lengths_test, mask_test, instance_ids=instance_ids, is_training=False, freeze_alpha=True)
                outputs_test = outputs_test.squeeze(1)

                preds_samples.append(torch.sigmoid(outputs_test).cpu().numpy())

            preds_samples = np.array(preds_samples)
            mean_pred = preds_samples.mean(axis=0)
            std_pred = preds_samples.std(axis=0)

            all_mean_preds.extend(mean_pred)
            all_std_preds.extend(std_pred)
            all_labels.extend(labels_test.cpu().numpy())
            all_probabilities.extend(mean_pred)

            for instance_id, mean_p, std_p in zip(instance_ids, mean_pred, std_pred):
                instance_uncertainty[instance_id] = {"mean_pred": mean_p, "std_pred": std_p}

    all_mean_preds = np.array(all_mean_preds)
    all_std_preds = np.array(all_std_preds)
    all_labels = np.array(all_labels)

    predicted_labels = (all_mean_preds > threshold).astype(int)
    accuracy = np.mean(predicted_labels == all_labels) * 100
    overall_uncertainty = all_std_preds.mean()

    roc_auc_test = round(roc_auc_score(all_labels, all_probabilities), 2)
    fpr_test, tpr_test, thresholds_test = roc_curve(all_labels, all_probabilities)

    print("\n Classification Report:")
    print(classification_report(all_labels, predicted_labels, target_names=["AI-generated", "Human-generated"]))
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print(f" Overall Model Uncertainty: {overall_uncertainty:.4f}")
    print(f" ROC-AUC Score: {roc_auc_test:.4f}")

    return accuracy, overall_uncertainty, fpr_test, tpr_test, roc_auc_test, instance_uncertainty

def test_performance_bayesian(testdata, model, criterion, threshold = 0.5, lstm=False):
    
    #print("Test performance")
    model.eval()
    test_loss = 0
    test_correct_predictions = 0
    test_total_predictions = 0
    
    all_labels = []
    all_predictions = []
    
    with torch.no_grad():
                 for inputs_test, labels_test, lengths_test, instance_ids in testdata:
                   mask_test = (inputs_test != pad_idx)
                   outputs_test_tuple = model(inputs_test, lengths_test, mask_test, instance_ids=instance_ids, is_training=False)
                   outputs_test = outputs_test_tuple[0]  
                   outputs_test = outputs_test.squeeze(1)
                   likelihood_test_loss = criterion(outputs_test, labels_test.float())
                   
                   kl_loss_test = kl.kl_divergence(
                       torch.distributions.Normal(model.bayes_fc.weight_mu, torch.exp(0.5*model.bayes_fc.weight_log_sigma)),
                       torch.distributions.Normal(model.bayes_fc.prior_mu, model.bayes_fc.prior_sigma)  
                   ).sum()
                   
                   loss = likelihood_test_loss + 0.01 * kl_loss_test  
                   test_loss += loss.item()
                   
                   predicted_labels_test = (torch.sigmoid(outputs_test) > threshold).float()
                   test_correct_predictions += (predicted_labels_test == labels_test).sum().item()
                   test_total_predictions += labels_test.size(0)
                   
                   all_labels.extend(labels_test.tolist())
                   all_predictions.extend(predicted_labels_test.tolist())
                   

    test_avg_loss = test_loss/len(test_dataloader)
    test_accuracy = test_correct_predictions / test_total_predictions
    print(f'Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_avg_loss:.4f}%')
    
    precision_test = round(precision_score(all_labels, all_predictions),2)
    recall_test = round(recall_score(all_labels, all_predictions),2)
    f1_test = round(f1_score(all_labels, all_predictions),2)
    roc_auc_test = round(roc_auc_score(all_labels, [p > 0.5 for p in all_predictions]),2)
    fpr_test, tpr_test, thresholds_test = roc_curve(all_labels, all_predictions)
    
    print(f"Precision: {precision_test}")
    print(f"Recall: {recall_test:.4f}")
    print(f"F1 Score: {f1_test:.4f}")
    print(f"ROC-AUC: {roc_auc_test:.4f}")
    
    return fpr_test, tpr_test, roc_auc_test



def compute_accuracy_intuiBayes(model, dataloader, threshold=0.5):
    model.eval()
    all_predictions = []
    all_labels = []
    with torch.no_grad():
        for inputs, labels, lengths in dataloader:
            mask = (inputs != pad_idx)
            outputs = model(inputs, lengths, mask)
            predictions = torch.sigmoid(outputs).squeeze(1)  
            all_predictions.extend(predictions.cpu().numpy())  
            all_labels.extend(labels.cpu().numpy())
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    predicted_labels = (all_predictions > threshold).astype(int)
    accuracy = np.mean(predicted_labels == all_labels) * 100

    print("Classification Report:")
    print(classification_report(all_labels, predicted_labels, target_names=["AI-generated", "Human-generated"]))

    return accuracy    

def test_performance(testdata, model, criterion, threshold = 0.5, lstm=False):
    
    print("Test performance")
    model.eval()
    test_loss = 0
    test_correct_predictions = 0
    test_total_predictions = 0
    all_labels = []
    all_predictions = []
    with torch.no_grad():
                 for inputs_test, labels_test, lengths_test in testdata:
                   mask_test = (inputs_test != pad_idx)
                   outputs_test = model(inputs_test, lengths_test, mask_test)
                   outputs_test = outputs_test.squeeze(1)
                   loss_test = criterion(outputs_test, labels_test.float())
                   test_loss += loss_test.item()
                   
                   predicted_labels_test = (torch.sigmoid(outputs_test) > threshold).float()
                   test_correct_predictions += (predicted_labels_test == labels_test).sum().item()
                   test_total_predictions += labels_test.size(0)
                   
                   all_labels.extend(labels_test.tolist())
                   all_predictions.extend(predicted_labels_test.tolist())
                   

    test_avg_loss = test_loss/len(test_dataloader)
    test_accuracy = test_correct_predictions / test_total_predictions
    print(f'Test Accuracy: {test_accuracy:.4f}, Test Loss: {test_avg_loss:.4f}%')
    
    precision_test = round(precision_score(all_labels, all_predictions),2)
    recall_test = round(recall_score(all_labels, all_predictions),2)
    f1_test = round(f1_score(all_labels, all_predictions),2)
    roc_auc_test = round(roc_auc_score(all_labels, [p > 0.5 for p in all_predictions]),2)
    fpr_test, tpr_test, thresholds_test = roc_curve(all_labels, all_predictions)
    
    print(f"Precision: {precision_test}")
    print(f"Recall: {recall_test:.4f}")
    print(f"F1 Score: {f1_test:.4f}")
    print(f"ROC-AUC: {roc_auc_test:.4f}")
    
    return fpr_test, tpr_test, roc_auc_test

def plot_metrics2(train_accuracies, val_accuracies, train_losses, val_losses, fpr_test, tpr_test, roc_auc_test):
    plt.figure(figsize=(11, 6))
    plt.plot(train_accuracies, label='Training Accuracy')
    plt.plot(val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Training and Validation Accuracy over Epochs')
    plt.ylim(0, 1)
    plt.legend()
    plt.show()

    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss over Epochs')
    plt.legend()
    plt.show()
    
    plt.figure()
    plt.plot(fpr_test, tpr_test, label='ROC curve (area=%0.2f)' % roc_auc_test)
    plt.plot([0,1], [0,1], 'k--') #Dashed diagonal
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.show()


def plot_metrics(train_accuracies, val_accuracies, train_losses, val_losses, fpr_test, tpr_test, roc_auc_test):
    epochs = range(1, len(train_accuracies) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    axes[0].plot(epochs, train_losses, label='Training Loss', color='orange', marker='o')
    axes[0].plot(epochs, val_losses, label='Validation Loss', color='orangered', marker='o')
    axes[0].set_title("Loss over Epochs")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].legend()
    axes[0].grid(True)

    axes[1].plot(epochs, train_accuracies, label='Training Accuracy', color='gold', marker='o')
    axes[1].plot(epochs, val_accuracies, label='Validation Accuracy', color='darkorange', marker='o')
    axes[1].set_title("Accuracy over Epochs")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(6, 5))
    plt.plot(fpr_test, tpr_test, label=f'ROC curve (area = {roc_auc_test:.2f})', color='darkblue')
    plt.plot([0, 1], [0, 1], 'k--', label='Random Guess')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.show()

fl1 =  'domain1_train.json' 
fl2 =  'domain2_train.json' 

visualise_data(file=fl2)
df2 = truncate_texts(fl2)
#print("Visualing domain 2 after truncating")
visualise_data(df=df2)

visualise_data(file=fl1)
df1 = truncate_texts(fl1)
#print("Visualing domain 1 after truncating")
visualise_data(df=df1)

train_df, val_df, test_df, _ =  split_data_d1(df1) 
train_df2, val_df2, test_df2, ratio = split_data_d2_augmented(df2)
#print("the unbalanced ratio is: ", ratio)

train_dataloader, val_dataloader, test_dataloader = generateDataLoader(train_df, val_df, test_df, batch_size = 500) 
train_dataloader2, val_dataloader2, test_dataloader2 = generateDataLoader(train_df2, val_df2, test_df2, batch_size = 50) 

def plot_alpha_distribution(model):

    if not hasattr(model, 'alpha_values') or len(model.alpha_values) == 0:
        print("No Alpha values recorded. Ensure that Alpha is stored during training.")
        return
    try:
        alpha_values = np.array(model.alpha_values, dtype=float).flatten()
    except ValueError:
        print("Error: Alpha values contain non-numeric data. Ensure they are properly stored as floats.")
        return
    if alpha_values.size == 0:
        print("Alpha values are empty after conversion. No plot will be generated.")
        return
    plt.figure(figsize=(8, 5))
    plt.hist(alpha_values, bins=30, color='blue', alpha=0.7, edgecolor='black')
    plt.axvline(np.mean(alpha_values), color='red', linestyle='dashed', linewidth=2, label=f'Mean Alpha: {np.mean(alpha_values):.4f}')
    plt.axvline(np.median(alpha_values), color='green', linestyle='dashed', linewidth=2, label=f'Median Alpha: {np.median(alpha_values):.4f}')
    plt.title("Alpha Distribution Analysis")
    plt.xlabel("Alpha Value")
    plt.ylabel("Frequency")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_retrieval_frequencies(memory_bank):
    """Plots retrieval frequency per cluster, not per instance."""
    
    if len(memory_bank.retrieval_counts) == 0:
        print("No memory retrievals recorded yet!")
        return

    retrieval_counts = Counter(memory_bank.retrieval_counts)

    print(f"Cluster Retrieval Counts: {retrieval_counts}")

    plt.figure(figsize=(8, 4))
    plt.bar(retrieval_counts.keys(), retrieval_counts.values(), color='blue')
    plt.xlabel("Memory Cluster")
    plt.ylabel("Retrieval Frequency")
    plt.title("Memory Retrieval Frequencies")
    plt.show()


def robot_IntuiBayes(domain='1'):
    
    
  if domain == '2':  
    embedding_dim=24
    hidden_dim=16
    n_layers = 1
    bidirectional = True
    dropout = 0.2
    attention= False
   
    criterion = torch.nn.BCEWithLogitsLoss() 
    num_epochs=6
    batch_size = 50
    lr=0.0075
    threshold = 0.5
    wl = ratio
    pos_weight = torch.tensor([wl])
    criterion_training = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)
   
    train, val, test = generateDataLoader(train_df2, val_df2, test_df2, batch_size = batch_size) 

  else:  
    embedding_dim=24
    hidden_dim=16
    n_layers = 1
    bidirectional = True
    dropout = 0
    attention= True
     
    criterion = torch.nn.BCEWithLogitsLoss() 
    criterion_training = torch.nn.BCEWithLogitsLoss() 
    num_epochs=3 
    batch_size = 250
    lr=0.0075
    threshold = 0.5
    
    train, val, test = generateDataLoader(train_df, val_df, test_df, batch_size = batch_size) 
    
  
  model_intuiBayes = BayesIntuit(vocab_size=vocab_size, embedding_dim= embedding_dim, hidden_dim=hidden_dim, num_layers=n_layers, bidirectional=bidirectional, dropout=dropout)
  
  
  print("Training IntuiBayes with data from Domain: ",domain)
  model_trained, train_accuracies, val_accuracies, train_losses, val_losses = bayesian_train(train, val, model_intuiBayes, criterion=criterion_training,
                                                                                       lr=lr, num_epochs =num_epochs, threshold = threshold)
  #print("IntuiBayes is predicting...")
  fpr_test, tpr_test, roc_auc_test = test_performance_bayesian(test, model_trained, criterion, lstm=True)
  plot_metrics(train_accuracies, val_accuracies, train_losses, val_losses, fpr_test, tpr_test, roc_auc_test)
  
  test_performance_bayesian_with_uncertainty(test, model_trained, criterion)
  plot_retrieval_frequencies(model_trained.memory_bank)
  plot_alpha_distribution(model_trained)
  memory_bank = model_trained.memory_bank
  memory_bank.compute_cluster_centroids()
  metrics = memory_bank.compute_intra_cluster_metrics()
  nearest_examples = memory_bank.find_nearest_examples_to_centroids(top_k=5)

  torch.save(memory_bank.cluster_centroids, "cluster_centroids.pt")

robot_IntuiBayes(domain='1')

#combined_output


