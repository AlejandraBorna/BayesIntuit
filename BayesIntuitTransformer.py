# -*- coding: utf-8 -*-
"""
Created on Sat May 24 17:43:05 2025

@author: Mayra Bornacelly
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
from collections import Counter
from torchvision import models
from sklearn.cluster import KMeans
import faiss  
from sklearn.cluster import KMeans
from torch.distributions.kl import kl_divergence  
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
from scipy.stats import iqr, spearmanr
import torch, numpy as np, random
import math
from collections import Counter
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

vocab_size = 5001 
pad_idx = 5000 
lambda_l1=0.01 

def print_tokenized_dataset_analysis(file=None, df=None):
    """
    Analyzes a tokenized dataset (from file or DataFrame) with elements
    (token_list, label), where tokens are integer indices from 0–4999 (0 = unknown).
    Prints statistics per class (AI = 1, Human = 0).
    """
    if df is None:
        if file is None:
            raise ValueError("Either a file or a DataFrame must be provided")
        data = []
        with open(file, 'r') as f:
            for line in f:
                data.append(json.loads(line))
        dataset = pd.DataFrame(data)
    else:
        dataset = df

    dataset['text_length'] = dataset['text'].apply(len)
    results = {}

    for label in [0, 1]:  # Human = 0, AI = 1
        token_seqs = dataset[dataset['label'] == label]['text'].tolist()
        lengths = [len(seq) for seq in token_seqs]

        max_len = max(lengths)
        length_stats = {
            "count": len(lengths),
            "mean_length": np.mean(lengths),
            "std_length": np.std(lengths),
            "min_length": np.min(lengths),
            "max_length": max_len,
            "pct_max_length": 100 * np.sum(np.array(lengths) == max_len) / len(lengths),
            "length_bin_counts": {
                "0–20": sum(0 <= l <= 20 for l in lengths),
                "21–50": sum(21 <= l <= 50 for l in lengths),
                "51–100": sum(51 <= l <= 100 for l in lengths),
                "101–200": sum(101 <= l <= 200 for l in lengths),
                "201–400": sum(201 <= l <= 400 for l in lengths),
            }
        }

        unique_token_ratios = []
        repeated_token_counts = []
        for seq in token_seqs:
            counter = Counter(seq)
            unique_count = len(counter)
            repeated_count = sum(1 for c in counter.values() if c > 1)
            unique_token_ratios.append(unique_count / len(seq) if len(seq) > 0 else 0)
            repeated_token_counts.append(repeated_count)

        diversity_stats = {
            "mean_unique_tokens": np.mean([len(set(seq)) for seq in token_seqs]),
            "mean_unique_token_ratio": np.mean(unique_token_ratios),
            "mean_repeated_token_count": np.mean(repeated_token_counts)
        }

        all_tokens = [token for seq in token_seqs for token in seq]
        token_freq = Counter(all_tokens)
        total = sum(token_freq.values())
        token_probs = [count / total for count in token_freq.values()]
        entropy = -sum(p * math.log(p + 1e-9) for p in token_probs)

        frequency_stats = {
            "top_10_tokens": token_freq.most_common(10),
            "token_freq_entropy": entropy
        }
        first_tokens = [seq[0] for seq in token_seqs if len(seq) > 0]
        last_tokens = [seq[-1] for seq in token_seqs if len(seq) > 0]
        first_freq = Counter(first_tokens)
        last_freq = Counter(last_tokens)

        pos_stats = {
            "top_5_first_tokens": first_freq.most_common(5),
            "top_5_last_tokens": last_freq.most_common(5),
            "first_token_entropy": -sum((c / len(first_tokens)) * math.log((c / len(first_tokens)) + 1e-9)
                                        for c in first_freq.values()),
            "last_token_entropy": -sum((c / len(last_tokens)) * math.log((c / len(last_tokens)) + 1e-9)
                                       for c in last_freq.values())
        }

        results[f"class_{label}"] = {
            "length_stats": length_stats,
            "diversity_stats": diversity_stats,
            "frequency_stats": frequency_stats,
            "positional_stats": pos_stats
        }

    for class_label, stats in results.items():
        print(f"\n====== Statistics for {class_label} ======")
        for stat_group, group_values in stats.items():
            print(f"\n--- {stat_group} ---")
            for key, value in group_values.items():
                if isinstance(value, dict):
                    for sub_key, sub_val in value.items():
                        print(f"{sub_key}: {sub_val}")
                else:
                    print(f"{key}: {value}")

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
    

    
def truncate_texts(fl, column='text_length', chunk_size=128, max_chunks=3):
    import json
    import pandas as pd

    data = []
    with open(fl, 'r') as file:
        for line in file:
            data.append(json.loads(line))
    df = pd.DataFrame(data)
    df['text_length'] = df['text'].apply(len)
    max_len = chunk_size * max_chunks

    def truncate_to_chunk(text):
        return text[:max_len] if len(text) > max_len else text
    df['text'] = df['text'].apply(truncate_to_chunk)
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
    print(f"Average length: {average_length}, Longest length: {longest_length}, Shortest length: {shortest_length}")
    print("AI text samples: ",len(df[df['label']==0])," human samples: ",len(df[df['label']==1]))
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
    print("Training set size:", len(train), "Validation set size:", len(val), "Test set size:", len(test))
    print("Training AI samples:", len(train[train['label'] == 0]), "Training human samples:", len(train[train['label'] == 1]), "Ratio:", ratio)
    print("Validation AI samples:", len(val[val['label'] == 0]), "Validation human samples:", len(val[val['label'] == 1]), "Ratio:", 
          round(len(val[val['label'] == 0]) / len(val[val['label'] == 1]), 2))
    print("Test AI samples:", len(test[test['label'] == 0]), "Test human samples:", len(test[test['label'] == 1]))
    return train, val, test, ratio


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

class AttentionPooling(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.Linear(dim, 1)

    def forward(self, x, mask=None):
        # x: [batch, seq_len, dim]
        attn_weights = self.attn(x).squeeze(-1)  # [batch, seq_len]
        if mask is not None:
            attn_weights = attn_weights.masked_fill(mask == 0, -1e9)
        attn_scores = torch.softmax(attn_weights, dim=1)  # [batch, seq_len]
        pooled = torch.bmm(attn_scores.unsqueeze(1), x).squeeze(1)  # [batch, dim]
        return pooled

class DynamicMemoryBank:
    def __init__(self, memory_dim, num_clusters=10, min_cluster_size=5,
                 memory_bank_size=8000, cluster_threshold=800):
        self.memory_dim = memory_dim
        self.num_clusters = num_clusters
        self.min_cluster_size = min_cluster_size
        self.memory_bank_size = memory_bank_size
        self.cluster_threshold = cluster_threshold
        self.memory_dict = {}  # instance_id → memory vector
        self.cluster_labels = {}  # instance_id → cluster label
        self.retrieval_counts = {i: 0 for i in range(num_clusters)}
        self.new_memory_count = 0  # counter to trigger re-clustering
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

    def retrieve(self, instance_id, query, is_training=True, current_epoch=None):
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
            # Fallback cluster-based retrieval
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

        if not hasattr(self, "memory_retrieval_debug_log"):
            self.memory_retrieval_debug_log = []

        self.memory_retrieval_debug_log.append({
            "epoch": current_epoch,
            "instance_id": instance_id,
            "retrieval_source": debug_info.get("retrieval_source"),
            "cosine_sim": debug_info.get("cos_sim"),
            "reliability": debug_info.get("reliability"),
            "query_norm": debug_info.get("query_norm"),
            "memory_norm": debug_info.get("memory_norm"),
            "cluster_id": self.cluster_labels.get(instance_id, -1),
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

    def update_memory(self):
        print(f"\n🔄 Updating memory bank at the end of Epoch {self.last_cluster_update_epoch + 1}...")
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

        print("Cluster centroids computed.")
        for k, v in cluster_centroids.items():
            print(f"Cluster {k} centroid norm: {v.norm().item():.4f}")
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
        # Cosine similarities
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

        print(f"Cluster {cluster_id}:")
        print(f"  - Members: {len(members)}")
        print(f"  - Mean distance to centroid: {mean_dist:.4f}")
        print(f"  - Variance of distances: {var_dist:.4f}")
        print(f"  - Mean cosine similarity: {mean_cos_sim:.4f}")
        print()

     return results
 
class DeepCluster(nn.Module):
    def __init__(self, embedding_dim, num_clusters=10):
        super(DeepCluster, self).__init__()
        self.num_clusters = num_clusters
        self.feature_extractor = nn.Sequential(
            nn.Linear(embedding_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 32),  
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
        """Predicts cluster assignment for a single query."""
        if self.kmeans is None:
            print(" KMeans not trained yet. Returning default cluster -1.")
            return torch.tensor(-1)

        query_np = query.detach().cpu().numpy().reshape(1, -1)
        query_np = query_np / (np.linalg.norm(query_np, axis=1, keepdims=True) + 1e-6)

        try:
            cluster_id = self.kmeans.predict(query_np)[0]
        except Exception as e:
            print(f" Cluster prediction failed: {e}")
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
           

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.pe = pe.unsqueeze(0)  

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :].to(x.device)

class basicTransformer(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256,
                 num_heads=4, num_layers=1, chunk_size=128, pad_idx=0):
        super().__init__()
        self.chunk_size = chunk_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len=chunk_size)
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.domain_embedding = nn.Embedding(2, embedding_dim)  
        self.fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, 1)

    def forward(self, text, text_lengths=None, mask=None, domain_flag=None,
                instance_ids=None, is_training=True, current_epoch=None, max_epochs=None, freeze_alpha=False):
        B, T = text.size()
        E = self.embedding.embedding_dim
        chunk_size = self.chunk_size
        num_chunks = (T + chunk_size - 1) // chunk_size  # round up
        pad_len = num_chunks * chunk_size - T
        if pad_len > 0:
            text = F.pad(text, (0, pad_len), value=self.embedding.padding_idx)
            if mask is not None:
                mask = F.pad(mask, (0, pad_len), value=0)
        embedded = self.embedding(text)  
        embedded = embedded.view(B * num_chunks, chunk_size, E)
        embedded = self.positional_encoding(embedded)
        if mask is not None:
            mask = mask.view(B * num_chunks, chunk_size)

        transformer_out = self.transformer_encoder(
            embedded,
            src_key_padding_mask=(~mask.bool()) if mask is not None else None
        )
        pooled_chunks = transformer_out.mean(dim=1)  
        pooled_chunks = pooled_chunks.view(B, num_chunks, E)
        pooled_output = pooled_chunks.mean(dim=1)  
        if domain_flag is not None:
            domain_emb = self.domain_embedding(domain_flag.long())  
            pooled_output = pooled_output + domain_emb
        x = F.relu(self.fc1(pooled_output))
        x = F.dropout(x, p=0.2, training=self.training)
        logits = self.output_layer(x)  
        return logits, None
    
class BayesIntuit(nn.Module):
    def __init__(self, vocab_size, embedding_dim=128, hidden_dim=256,
                 num_heads=4, num_layers=1, chunk_size=128, pad_idx=5000, max_seq_len=400, memory_bank_size=50,
                 num_clusters=10, projected_dim=None, use_alpha_in_output = True, use_memory=True, 
                 use_clipped_alpha_in_prediction=True, trust_memory_percent = 0.2):
        super().__init__()
        self.chunk_size = chunk_size
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_idx)
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len=chunk_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embedding_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.domain_embedding = nn.Embedding(2, embedding_dim)  
        self.fc1 = nn.Linear(embedding_dim, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, 1)
        self.use_projection = projected_dim is not None
        self.projector = nn.Linear(embedding_dim, projected_dim) if self.use_projection else None
        memory_dim = projected_dim if self.use_projection else embedding_dim
        self.memory_bank = DynamicMemoryBank(memory_dim=memory_dim, num_clusters=num_clusters)
        self.bayes_fc = bnn.BayesLinear(prior_mu=0, prior_sigma=0.1,
                                        in_features=memory_dim, out_features=1)
        self.alpha_head = HybridAlphaHead(input_dim=memory_dim)
        self.context_evaluation = None  
        self.alpha_values = []
        self.cosine_similarities = []
        self.reliabilities = []  
        self.use_alpha_in_output = use_alpha_in_output 
        self.use_memory = use_memory
        if self.use_memory == False and self.use_alpha_in_output == True:
           self.use_memory = True
        self.use_clipped_alpha_in_prediction = use_clipped_alpha_in_prediction
        self.trust_memory_percent = trust_memory_percent 
        self.alpha_1 = nn.Parameter(torch.tensor(10.0), requires_grad=True)
        self.alpha_2 = nn.Parameter(torch.tensor(40.0), requires_grad=True)
        self.debug_memory_retrieval_logs = []
        self.max_debug_logs = 25  

    def forward(self, text, text_lengths=None, mask=None, domain_flag=None,
                instance_ids=None, is_training=True, current_epoch=None, max_epochs=None, freeze_alpha=False):
        
        B, T = text.size()
        E = self.embedding.embedding_dim
        chunk_size = self.chunk_size
        num_chunks = (T + chunk_size - 1) // chunk_size  

        pad_len = num_chunks * chunk_size - T
        if pad_len > 0:
            text = F.pad(text, (0, pad_len), value=self.embedding.padding_idx)
            if mask is not None:
                mask = F.pad(mask, (0, pad_len), value=0)

        embedded = self.embedding(text)  
        embedded = embedded.view(B * num_chunks, chunk_size, E)
        embedded = self.positional_encoding(embedded)

        if mask is not None:
            mask = mask.view(B * num_chunks, chunk_size)

        transformer_out = self.transformer_encoder(
            embedded,
            src_key_padding_mask=(~mask.bool()) if mask is not None else None
        )
        pooled_chunks = transformer_out.mean(dim=1)  
        pooled_chunks = pooled_chunks.view(B, num_chunks, E)
        pooled_output = pooled_chunks.mean(dim=1)  

        if domain_flag is not None:
            domain_emb = self.domain_embedding(domain_flag.long())  
            pooled_output = pooled_output + domain_emb

        x = F.relu(self.fc1(pooled_output))
        x = F.dropout(x, p=0.2, training=self.training)
        
        attn_output_processed = self.apply_projection(x)
        
        # === Memory Retrieval ===
        if self.use_memory:
           retrieved_memories, reliabilities, retrieval_sources = [], [], []
     
           for i in range(attn_output_processed.size(0)):
              instance_id = instance_ids[i]
              result = self.memory_bank.retrieve(instance_id, attn_output_processed[i], is_training=is_training)

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
             alpha = alpha_mean.view(-1, 1)  
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

        if hasattr(self, "alpha_values"):
           self.alpha_values.extend(alpha_mean.detach().cpu().tolist())
        if hasattr(self, "reliability_values"):
           self.reliability_values.extend(reliabilities.detach().cpu().numpy().flatten().tolist())
        
        # === Blend Alpha with Reliability
        alpha = alpha * reliabilities + (1 - reliabilities) * 0.5
        alpha = torch.clamp(alpha, 0.01, 0.99)
        
        #Informative Alpha
        alpha_informative = alpha_informative * reliabilities + (1 - reliabilities) * 0.5
        alpha_informative = torch.clamp(alpha_informative, 0.01, 0.99)
        
        if not self.use_alpha_in_output:
           alpha = torch.zeros_like(alpha)

         
        if self.use_memory:
           alpha_clipped = torch.clamp(alpha, min=0.01, max=0.99) #0.3

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
        combined_output = attn_output_processed + (alpha_clipped * retrieved_memory)# Additively
        #combined_output = (attn_output_processed * (1-alpha_clipped)) + (alpha_clipped * retrieved_memory) #Convex combinatory
        dense_outputs = self.bayes_fc(combined_output) 

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
       
    def apply_projection(self, tensor):
        if self.use_projection and self.projector is not None:
                 return self.projector(tensor)
        return tensor  

def bayesian_train_chunked(traindata, valdata, model, domain_flag=1, criterion=torch.nn.BCEWithLogitsLoss(),
                           lr=0.001, num_epochs=3, WithAttention=True, threshold=0.5, patience=4, pad_idx=0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []
    
    alpha_means_epoch, alpha_vars_epoch = [], []
    reliability_means_epoch, reliability_vars_epoch = [], []
    alpha1_means_epoch, alpha2_means_epoch = [], []

    best_val_loss = float('inf')
    epochs_no_improve = 0

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
            domain_flag_batch = torch.full((inputs_train.size(0),), domain_flag, dtype=torch.long, device=inputs_train.device)

            outputs, alpha_entropy, debug_info = model(inputs_train, lengths_train, mask_train,
                               domain_flag=domain_flag_batch,
                               instance_ids=instance_ids, is_training=True,
                               current_epoch=epoch, max_epochs=num_epochs)
            outputs = outputs.squeeze(1)
            likelihood_loss = criterion(outputs, labels_train.float())
            kl_loss = kl.kl_divergence(
                torch.distributions.Normal(model.bayes_fc.weight_mu, torch.exp(0.5*model.bayes_fc.weight_log_sigma)),
                torch.distributions.Normal(model.bayes_fc.prior_mu, model.bayes_fc.prior_sigma)  
            ).sum()
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

            loss = likelihood_loss + 0.01 * kl_loss  
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
                domain_flag_batch = torch.full((inputs_val.size(0),), domain_flag, dtype=torch.long, device=inputs_val.device)

                outputs_val, _ , debug_info = model(inputs_val, lengths_val, mask_val,
                                       domain_flag=domain_flag_batch,
                                       instance_ids=instance_ids, is_training=False,
                                       current_epoch=epoch, max_epochs=num_epochs)
                outputs_val = outputs_val.squeeze(1)
                likelihood_val_loss = criterion(outputs_val, labels_val.float())
                
                kl_loss_val = kl.kl_divergence(
                    torch.distributions.Normal(model.bayes_fc.weight_mu, torch.exp(0.5*model.bayes_fc.weight_log_sigma)),
                    torch.distributions.Normal(model.bayes_fc.prior_mu, model.bayes_fc.prior_sigma)  # Add the prior distribution
                ).sum()
                
                loss = likelihood_val_loss + 0.00 * kl_loss_val  # Weighted KL term
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
        print(f"  Losses -> Likelihood: {loss.item():.4f}")
        print(f'Epoch {epoch+1}: Training Accuracy: {accuracy:.4f}%, Training Loss: {average_loss:.4f}%, '
              f'Validation Accuracy: {val_accuracy:.4f}%, Validation Loss: {val_avg_loss:.4f}%, '
              f'Validation ROC-AUC: {roc_auc_val:.4f}%')
        print(f"  Reliability Mean: {np.mean(model.reliability_values):.4f}")
        print(f"📈 Alpha Mean (Epoch {epoch+1}): {alpha_array.mean():.4f}, Alpha Var: {alpha_array.var():.4f}, "
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

      N = 2000
      logs_df = pd.DataFrame(model.debug_memory_retrieval_logs[:N])
      logs_df.to_csv("memory_retrieval_debug_TRANSFORMERS.csv", index=False)
      print(logs_df)
    
    if hasattr(model, "memory_bank") and hasattr(model.memory_bank, "memory_retrieval_debug_log"):
      df_log = pd.DataFrame(model.memory_bank.memory_retrieval_debug_log)
      df_log.to_csv("retrieval_source_Transformers.csv", index=False)
      print("✅ Memory retrieval log saved to memory_retrieval_trace_domain2.csv")

    return model, train_accuracies, val_accuracies, train_losses, val_losses


def plot_metrics(train_accuracies, val_accuracies, train_losses, val_losses, fpr_test, tpr_test, roc_auc_test):
    epochs = range(1, len(train_accuracies) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Loss Plot
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

def plot_retrieval_frequencies(memory_bank):
    """Plots retrieval frequency per cluster, not per instance."""
    
    if len(memory_bank.retrieval_counts) == 0:
        print("No memory retrievals recorded yet!")
        return


    # ✅ Aggregate retrieval counts by cluster
    retrieval_counts = Counter(memory_bank.retrieval_counts)

    print(f"📊 Cluster Retrieval Counts: {retrieval_counts}")

    plt.figure(figsize=(8, 4))
    plt.bar(retrieval_counts.keys(), retrieval_counts.values(), color='blue')
    plt.xlabel("Memory Cluster")
    plt.ylabel("Retrieval Frequency")
    plt.title("Memory Retrieval Frequencies")
    plt.show()

def plot_alpha_evolution(model):
    """
    Plots the evolution of Alpha values over training steps.
    Args:
        model: The trained BayesBelle model with alpha_values recorded during training.
    """
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

    plt.figure(figsize=(10, 5))
    plt.plot(alpha_values, color='blue', linewidth=1.5, label='Alpha over steps')
    plt.axhline(np.mean(alpha_values), color='red', linestyle='--', linewidth=1.5, label=f'Mean Alpha: {np.mean(alpha_values):.4f}')
    plt.axhline(np.median(alpha_values), color='green', linestyle='--', linewidth=1.5, label=f'Median Alpha: {np.median(alpha_values):.4f}')
    plt.title("Alpha Evolution Over Training Steps")
    plt.xlabel("Training Step")
    plt.ylabel("Alpha Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def bayesian_train_basic(traindata, valdata, model, domain_flag=1, criterion=torch.nn.BCEWithLogitsLoss(),
                           lr=0.001, num_epochs=3, WithAttention=True, threshold=0.5, patience=4, pad_idx=0):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    train_accuracies = []
    val_accuracies = []
    train_losses = []
    val_losses = []
    
    best_val_loss = float('inf')
    epochs_no_improve = 0

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
            domain_flag_batch = torch.full((inputs_train.size(0),), domain_flag, dtype=torch.long, device=inputs_train.device)

            outputs, _ = model(inputs_train, lengths_train, mask_train,
                               domain_flag=domain_flag_batch,
                               instance_ids=instance_ids, is_training=True,
                               current_epoch=epoch, max_epochs=num_epochs)
            outputs = outputs.squeeze(1)
            likelihood_loss = criterion(outputs, labels_train.float())
            loss = likelihood_loss 
  
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
                domain_flag_batch = torch.full((inputs_val.size(0),), domain_flag, dtype=torch.long, device=inputs_val.device)

                outputs_val, _ = model(inputs_val, lengths_val, mask_val,
                                       domain_flag=domain_flag_batch,
                                       instance_ids=instance_ids, is_training=False,
                                       current_epoch=epoch, max_epochs=num_epochs)
                outputs_val = outputs_val.squeeze(1)
                likelihood_val_loss = criterion(outputs_val, labels_val.float())
              
                
                loss = likelihood_val_loss 
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

        print(f"[Epoch {epoch+1}]")
        print(f"  Losses -> Likelihood: {loss.item():.4f}")
        print(f'Epoch {epoch+1}: Training Accuracy: {accuracy:.4f}%, Training Loss: {average_loss:.4f}%, '
              f'Validation Accuracy: {val_accuracy:.4f}%, Validation Loss: {val_avg_loss:.4f}%, '
              f'Validation ROC-AUC: {roc_auc_val:.4f}%')
       
        if val_avg_loss < best_val_loss:
            best_val_loss = val_avg_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if epochs_no_improve == patience:
            print("Early stopping triggered.")
            break

    return model, train_accuracies, val_accuracies, train_losses, val_losses

fl1 =  'domain1_train.json' 
fl2 =  'domain2_train.json' 

visualise_data(file=fl2)
df2 = truncate_texts(fl2)
print("Visualing domain 2 after truncating")
visualise_data(df=df2)

#visualise_data(file=fl1)
df1 = truncate_texts(fl1)
print("Visualing domain 1 after truncating")
#visualise_data(df=df1)

train_df, val_df, test_df, _ =  split_data_d1(df1) 
train_df2, val_df2, test_df2, ratio = split_data_d2_augmented(df2)
#print("the unbalanced ratio is: ", ratio)

print_tokenized_dataset_analysis(df=df2)

train_dataloader, val_dataloader, test_dataloader = generateDataLoader(train_df, val_df, test_df, batch_size = 500) 
train_dataloader2, val_dataloader2, test_dataloader2 = generateDataLoader(train_df2, val_df2, test_df2, batch_size = 50) 

def test_performance_chunked_with_uncertainty(testdata, model, criterion, threshold=0.5, n_samples=20, pad_idx=0, domain_flag=1):
    print("Testing IntuiBayes with Uncertainty Estimation...")
    model.eval()

    all_mean_preds = []
    all_std_preds = []
    all_labels = []
    all_probabilities = []
    instance_uncertainty = {}

    with torch.no_grad():
        for batch_idx, (inputs_test, labels_test, lengths_test, instance_ids) in enumerate(testdata):
            mask_test = (inputs_test != pad_idx)
            domain_flag_batch = torch.full((inputs_test.size(0),), domain_flag, dtype=torch.long, device=inputs_test.device)

            preds_samples = []
            for _ in range(n_samples):
                outputs_test, _ , debug_info = model(
                    inputs_test, lengths_test, mask_test,
                    domain_flag=domain_flag_batch,
                    instance_ids=instance_ids,
                    is_training=False,
                    freeze_alpha=True
                )
                outputs_test = outputs_test.squeeze(1)
                preds_samples.append(torch.sigmoid(outputs_test).cpu().numpy())

            preds_samples = np.array(preds_samples)  # [n_samples, batch_size]
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

    roc_auc_test = round(roc_auc_score(all_labels, all_probabilities), 4)
    fpr_test, tpr_test, thresholds_test = roc_curve(all_labels, all_probabilities)

    print("\n🔹 Classification Report:")
    print(classification_report(all_labels, predicted_labels, target_names=["AI-generated", "Human-generated"]))
    print(f"\nTest Accuracy: {accuracy:.2f}%")
    print(f"Overall Model Uncertainty: {overall_uncertainty:.4f}")
    print(f"ROC-AUC Score: {roc_auc_test:.4f}")

    return accuracy, overall_uncertainty, fpr_test, tpr_test, roc_auc_test, instance_uncertainty


def robot_IntuiBayes(domain='1'):
 
  if domain == '1':
        domain_flag = 1  # easy domain
  elif domain == '2':
        domain_flag = 0  # complex domain
  else:
        raise ValueError("Invalid domain. Use '1' or '2'.")
  
  if domain == '2':  
    embedding_dim=24
    hidden_dim=16
    n_layers = 1
    bidirectional = True
    dropout = 0.2
    attention= False
    num_heads = 4
   
    criterion = torch.nn.BCEWithLogitsLoss() 
    num_epochs=10
    batch_size = 50
    lr=0.001
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
    num_epochs=6 
    batch_size = 250
    lr=0.001
    threshold = 0.5
    num_heads = 4
    
    train, val, test = generateDataLoader(train_df, val_df, test_df, batch_size = batch_size) 

  #model = BayesIntuit(vocab_size=vocab_size, embedding_dim= 128, hidden_dim=256, num_heads=4, num_layers=1, chunk_size=128, pad_idx=5000)
  model = basicTransformer(vocab_size=vocab_size, embedding_dim= 128, hidden_dim=256, num_heads=4, num_layers=1, chunk_size=128, pad_idx=5000)

 
  #print("Training IntuiBayes with data from Domain:", domain)
  model_trained, train_accuracies, val_accuracies, train_losses, val_losses = bayesian_train_chunked(
        train, val, model, domain_flag=domain_flag, criterion=criterion_training,
        lr=lr, num_epochs=num_epochs, threshold=threshold
    )
  """
  model_trained, train_accuracies, val_accuracies, train_losses, val_losses = bayesian_train_basic(
        train, val, model, domain_flag=domain_flag, criterion=criterion_training,
        lr=lr, num_epochs=num_epochs, threshold=threshold
    )
  """
  print("IntuiBayes is predicting...")
  accuracy, overall_uncertainty, fpr_test, tpr_test, roc_auc_test, instance_uncertainty = test_performance_chunked_with_uncertainty(
      test, model_trained, criterion, domain_flag=domain_flag
  )

  plot_metrics(train_accuracies, val_accuracies, train_losses, val_losses, fpr_test, tpr_test, roc_auc_test)
  memory_bank = model_trained.memory_bank
  memory_bank.compute_cluster_centroids()
  metrics = memory_bank.compute_intra_cluster_metrics()
  nearest_examples = memory_bank.find_nearest_examples_to_centroids(top_k=5)
  torch.save(memory_bank.cluster_centroids, "cluster_centroids.pt")

robot_IntuiBayes(domain='1')




