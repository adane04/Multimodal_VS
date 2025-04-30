import torch
from torch.utils.data import Dataset, DataLoader
import h5py
import json
import os
import numpy as np
from torch.nn.utils.rnn import pad_sequence

# Default paths
DEFAULT_VISUAL_PATH = './dataset/train_h5/features_with_text.h5'
DEFAULT_SPLIT_PATH = './dataset/train_h5/splits/'

class VideoData(Dataset):
    def __init__(self, name, mode, split_index, visual_path=DEFAULT_VISUAL_PATH):
        self.mode = mode
        self.name = name
        self.visual_path = visual_path
        self.split_file = f'{DEFAULT_SPLIT_PATH}{self.name}_splits.json'

        # Load split info
        with open(self.split_file, 'r') as f:
            data = json.load(f)
            self.split = data[split_index]

        # Load video names for this split
        self.video_names = self.split[self.mode + '_keys']

        # Load visual + text features from CLIP encoder
        self.visual_data = {}
        self.text_data = {}
        with h5py.File(self.visual_path, 'r') as hdf:
            for name in self.video_names:
                self.visual_data[name] = torch.tensor(np.array(hdf[name + '/features']), dtype=torch.float32)
                self.text_data[name] = torch.tensor(np.array(hdf[name + '/text_features']), dtype=torch.float32)

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, index):
        video_name = self.video_names[index]
        video_feats = self.visual_data[video_name]            # [T_v, 512]
        text_feats = self.text_data[video_name]               # [T_t, 512]
        return video_feats, text_feats, video_name

def collate_fn(batch):
    # Unpack all items
    video_seqs = []
    text_seqs = []
    video_names = []
    
    for item in batch:
        video_seqs.append(item[0])
        text_seqs.append(item[1])
        video_names.append(item[2])

    # Padding sequences
    padded_videos = pad_sequence(video_seqs, batch_first=True)  # [B, T_v_max, 512]
    padded_texts = pad_sequence(text_seqs, batch_first=True)    # [B, T_t_max, 512]

    # Create binary masks (1 for valid positions, 0 for padded)
    video_mask = torch.tensor([[1]*len(seq) + [0]*(padded_videos.size(1) - len(seq)) for seq in video_seqs])
    text_mask = torch.tensor([[1]*len(seq) + [0]*(padded_texts.size(1) - len(seq)) for seq in text_seqs])

    return padded_videos, padded_texts, {
        'video_mask': video_mask,     # [B, T_v_max]
        'text_mask': text_mask,       # [B, T_t_max]
        'video_names': video_names
    }

def get_loader(name, mode, split_index, batch_size=1, shuffle=False,
               visual_path=DEFAULT_VISUAL_PATH):
    
    dataset = VideoData(name, mode, split_index, visual_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return loader

class CustomVideoData(Dataset):
    def __init__(self, mode, visual_path, text_path):
        self.mode = mode
        self.visual_path = visual_path
        self.text_path = text_path

        # Load visual features
        hdf = h5py.File(self.visual_path, 'r')
        self.keys = list(hdf.keys())
        self.visual_data = {}
        for key in self.keys:
            self.visual_data[key] = torch.tensor(np.array(hdf[key + '/features']), dtype=torch.float32)
        hdf.close()

        # Load text features
        assert os.path.exists(self.text_path), f"File not found: {self.text_path}"
        self.text_features_dict = np.load(self.text_path, allow_pickle=True).item()

        # Filter keys 
        self.keys = [key for key in self.keys if key in self.text_features_dict]

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        video_name = self.keys[index]
        video_feats = self.visual_data[video_name]                     # [T_v, 1024]
        text_feats = torch.tensor(self.text_features_dict[video_name], dtype=torch.float32)  # [T_t, 768]

        if self.mode == 'test':
            return video_feats, text_feats, video_name
        else:
            return video_feats, text_feats
def get_loader_inference(mode, visual_path, text_path, batch_size=1, shuffle=False):
    dataset = CustomVideoData(mode, visual_path, text_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=collate_fn)
    return loader

