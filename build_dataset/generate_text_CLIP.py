# generate_text_CLIP.py generates textaul features
import numpy as np
import torch
import os
from typing import Union, List, Dict
from pathlib import Path
import clip  # Make sure clip is installed
from PIL import Image

class TextEmbedder:
    def __init__(self, device: str = None):
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model, _ = clip.load("ViT-B/32", device=self.device)

    def generate_embeddings(
        self,
        captions_path: str
    ) -> Dict[str, np.ndarray]:
    
        captions = {}
    
        if os.path.isfile(captions_path):
            # Single file
            video_id = Path(captions_path).stem
            with open(captions_path, 'r', encoding='utf-8') as f:
                captions[video_id] = f.read().strip().split('\n')
        else:
            # Folder with multiple .txt
            for file in os.listdir(captions_path):
                if file.endswith('.txt'):
                    video_id = Path(file).stem
                    with open(os.path.join(captions_path, file), 'r', encoding='utf-8') as f:
                        captions[video_id] = f.read().strip().split('\n')
    
        # Generate embeddings
        results = {}
        for idx, (video_id, texts) in enumerate(captions.items()):
            tokens = clip.tokenize(texts).to(self.device)
            with torch.no_grad():
                embeddings = self.model.encode_text(tokens)
                embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)  # Normalize
            
            # 
            save_key = f"video_{idx+1}"  # instead of using real video_id
            results[save_key] = embeddings.cpu().numpy()
    
        return results

if __name__ == "__main__":
    pass
    # captions_folder = "/path/to/test_query.txt"  # or folder
    # save_file = '/path/to/clip_text_embeddings.npy'
    # text_embedder = TextEmbedder()  # or TextEmbedder('cpu')
    # embeddings_dict = text_embedder.generate_embeddings(captions_folder)

    # np.save(save_file, embeddings_dict)
    # print(f"Saved embeddings to {save_file}")
