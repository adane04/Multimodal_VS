## this code generates visual features
import os
import numpy as np
import h5py
import torch
from decord import VideoReader
from tqdm import tqdm
from PIL import Image
import clip  # Make sure installed via pip
import decord

class GenerateDataset:
    def __init__(self, video_path, save_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)
        self.dataset = {}
        self.video_list = []
        self.video_path = ''
        self.h5_file = h5py.File(save_path, 'w')

        self.set_video_list(video_path)

    def set_video_list(self, video_path):
        if os.path.isdir(video_path):
            self.video_path = video_path
            self.video_list = sorted(os.listdir(video_path))
            self.video_list = [x for x in self.video_list if x.endswith(('.mp4', '.avi', '.mkv', '.mov'))]
        else:
            self.video_path = ''
            self.video_list.append(video_path)

        for idx, file_name in enumerate(self.video_list):
            self.dataset['video_{}'.format(idx + 1)] = {}
            self.h5_file.create_group('video_{}'.format(idx + 1))

    def extract_feature(self, frame):
        '''
        Extract frame feature using CLIP vision encoder.
        '''
        frame_img = Image.fromarray(frame)
        frame_tensor = self.preprocess(frame_img).unsqueeze(0).to(self.device)

        with torch.no_grad():
            features = self.clip_model.encode_image(frame_tensor)
            features = features / features.norm(dim=-1, keepdim=True)  # Normalize

        return features.squeeze(0).cpu().numpy()

    def get_change_points(self, video_path):
        '''
        Extract indices of keyframes using decord then construct segments.
        '''
        vr = VideoReader(video_path)
        fps = int(vr.get_avg_fps())
        n_frames = len(vr)

        key_indices = vr.get_key_indices()

        prev = int()
        key_indices_reduced = []
        for v in key_indices:
            if v - prev > fps:
                prev = v
                key_indices_reduced.append(v)

        key_indices_reduced = [0] + key_indices_reduced + [n_frames]

        temp_change_points = []
        for idx in range(len(key_indices_reduced) - 1):
            segment = [key_indices_reduced[idx], key_indices_reduced[idx + 1] - 1]
            temp_change_points.append(segment)

        change_points = np.array(temp_change_points)
        n_frame_per_seg = np.array([seg[1] - seg[0] + 1 for seg in change_points])

        return change_points, n_frame_per_seg

    def generate_dataset(self):
        '''
        Convert from video file (mp4) to h5 file with the right format.
        '''
        for video_idx, video_filename in enumerate(tqdm(self.video_list, desc='Feature Extract', ncols=80, leave=True)):
            video_path = video_filename
            if os.path.isdir(self.video_path):
                video_path = os.path.join(self.video_path, video_filename)

            video_name = os.path.basename(video_path)
            vr = VideoReader(video_path, width=224, height=224)  # CLIP model expects 224x224

            fps = vr.get_avg_fps()
            n_frames = len(vr)

            video_feat = None
            picks = []
            change_points, n_frame_per_seg = self.get_change_points(video_path)

            for segment in change_points:
                mid = (segment[0] + segment[1]) // 2
                frame = vr[mid].asnumpy()

                frame_feat = self.extract_feature(frame)
                picks.append(mid)

                if video_feat is None:
                    video_feat = frame_feat
                else:
                    video_feat = np.vstack((video_feat, frame_feat))

            self.h5_file['video_{}'.format(video_idx + 1)]['features'] = list(video_feat)
            self.h5_file['video_{}'.format(video_idx + 1)]['picks'] = np.array(list(picks))
            self.h5_file['video_{}'.format(video_idx + 1)]['n_frames'] = n_frames
            self.h5_file['video_{}'.format(video_idx + 1)]['fps'] = fps
            self.h5_file['video_{}'.format(video_idx + 1)]['change_points'] = change_points
            self.h5_file['video_{}'.format(video_idx + 1)]['n_frame_per_seg'] = n_frame_per_seg
            self.h5_file['video_{}'.format(video_idx + 1)]['video_name'] = video_name

        self.h5_file.close()

if __name__ == '__main__':
    pass
