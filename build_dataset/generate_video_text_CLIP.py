

#!pip install git+https://github.com/openai/CLIP.git


# generate_dataset_with_text.py
import os
import numpy as np
import h5py
import torch
from decord import VideoReader
from tqdm import tqdm
from PIL import Image
import clip
import decord
from transformers import BlipProcessor, BlipForConditionalGeneration
from TransNetV2_master.inference.transnetv2 import TransNetV2
import clip
import decord
from transformers import BlipProcessor, BlipForConditionalGeneration

class GenerateDataset:
    def __init__(self, video_path, save_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.clip_model, self.preprocess = clip.load("ViT-B/32", device=self.device)

        #  BLIP captioning model
        self.caption_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
        self.caption_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(self.device)

        self.dataset = {}
        self.video_list = []
        self.video_path = ''
        self.h5_file = h5py.File(save_path, 'w')
        self.model = TransNetV2()  # Initialize TransNetV2

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
        frame_img = Image.fromarray(frame)
        frame_tensor = self.preprocess(frame_img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.clip_model.encode_image(frame_tensor)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy()

    def generate_caption(self, frame):
        frame_img = Image.fromarray(frame)
        inputs = self.caption_processor(frame_img, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.caption_model.generate(**inputs)
            caption = self.caption_processor.decode(outputs[0], skip_special_tokens=True)
        return caption

    def encode_caption(self, caption_text):
        with torch.no_grad():
            tokens = clip.tokenize([caption_text]).to(self.device)
            features = self.clip_model.encode_text(tokens)
            features = features / features.norm(dim=-1, keepdim=True)
        return features.squeeze(0).cpu().numpy()

    def get_change_points(self, video_path):
        '''
        Extract indices of keyframes using TransNetV2 and construct segments
        '''
        # Get predictions from TransNetV2
        _, single_frame_predictions, _ = self.model.predict_video(video_path)
        shots = self.model.predictions_to_scenes(single_frame_predictions)

        # Convert scenes to change points
        change_points = np.array(shots)

        # Calculate the number of frames per segment
        n_frame_per_seg = np.array([end - start + 1 for start, end in change_points])

        return change_points, n_frame_per_seg

    def generate_dataset(self):
        '''
        Convert from video file (mp4) to h5 file with visual features, text features, and captions.
        '''
        for video_idx, video_filename in enumerate(tqdm(self.video_list, desc='Feature Extract', ncols=80, leave=True)):
            video_path = video_filename
            if os.path.isdir(self.video_path):
                video_path = os.path.join(self.video_path, video_filename)

            video_name = os.path.basename(video_path)
            vr = VideoReader(video_path, width=224, height=224)  # CLIP expects 224x224

            fps = vr.get_avg_fps()
            n_frames = len(vr)

            visual_feats = []
            text_feats = []
            captions = []
            picks = []
            change_points, n_frame_per_seg = self.get_change_points(video_path)

            for segment in change_points:
                mid = (segment[0] + segment[1]) // 2
                frame = vr[mid].asnumpy()

                frame_feat = self.extract_feature(frame)
                caption = self.generate_caption(frame)
                caption_feat = self.encode_caption(caption)

                visual_feats.append(frame_feat)
                text_feats.append(caption_feat)
                captions.append(caption)
                picks.append(mid)

            # Save to H5
            self.h5_file['video_{}'.format(video_idx + 1)]['video_features'] = np.array(visual_feats)
            self.h5_file['video_{}'.format(video_idx + 1)]['text_features'] = np.array(text_feats)
            self.h5_file['video_{}'.format(video_idx + 1)]['captions'] = np.array(captions, dtype=object)
            self.h5_file['video_{}'.format(video_idx + 1)]['picks'] = np.array(picks)
            self.h5_file['video_{}'.format(video_idx + 1)]['n_frames'] = n_frames
            self.h5_file['video_{}'.format(video_idx + 1)]['fps'] = fps
            self.h5_file['video_{}'.format(video_idx + 1)]['change_points'] = change_points
            self.h5_file['video_{}'.format(video_idx + 1)]['n_frame_per_seg'] = n_frame_per_seg
            self.h5_file['video_{}'.format(video_idx + 1)]['video_name'] = video_name

        self.h5_file.close()

if __name__ == '__main__':
    pass

#video_folder = "/path/TVSum/tvsum_video/"  # Path to your folder of input videos
#save_h5_path = "/path/output_features/features_with_text_TNet.h5"  # Where you want to save

#os.makedirs(os.path.dirname(save_h5_path), exist_ok=True)

#generator = GenerateDataset(video_folder, save_h5_path)
#generator.generate_dataset()
#print(f"Finished! Saved everything to: {save_h5_path}")





