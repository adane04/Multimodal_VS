import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import numpy as np
import torch.nn.functional as F
import json
import os
from tqdm import tqdm, trange
from models_train.layers.summarizer_multimodal import Summarizer
from models_train.layers.discriminator import Discriminator
from models_train.utils import TensorboardWriter

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

original_label = torch.tensor(1.0).to(device=device)
summary_label = torch.tensor(0.0).to(device=device)

class Solver(object):
    def __init__(self, config=None, train_loader=None, test_loader=None, margin=1.0):
        """Class that Builds, Trains and Evaluates SUM-GAN model"""
        self.config = config
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.n_epoch_trained = 0
        self.margin = margin

    def build(self):

        self.summarizer = Summarizer(input_size=self.config.hidden_size, 
                                   hidden_size=self.config.hidden_size, 
                                   num_layers=self.config.num_layers).to(device=device)
        
        self.discriminator = Discriminator(input_size=self.config.hidden_size, 
                                         hidden_size=self.config.hidden_size, 
                                         num_layers=self.config.num_layers).to(device=device)
        
        self.model = nn.ModuleList([
                                  self.summarizer, 
                                  self.discriminator])
     
        if self.config.mode == 'train':
            # Build Optimizers
            self.s_e_optimizer = optim.Adam(
                list(self.summarizer.attn.parameters()) +
                list(self.summarizer.vae.e_lstm.parameters()),
                 lr=self.config.lr
            )

            self.d_optimizer = optim.Adam(
                list(self.summarizer.vae.d_lstm.parameters()), 
                lr=self.config.lr
            )

            self.c_optimizer = optim.Adam(
                list(self.discriminator.parameters()),
                lr=self.config.discriminator_lr
            )

            self.model.train()
            self.writer = TensorboardWriter(self.config.log_dir)

    def loadfrom_checkpoint(self, path):
        checkpoint = torch.load(path, map_location=device)
        self.n_epoch_trained = checkpoint['n_epoch_trained']
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if self.config.mode == 'train':
            self.e_optimizer.load_state_dict(checkpoint['e_optimizer_state_dict'])
            self.d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
            self.c_optimizer.load_state_dict(checkpoint['c_optimizer_state_dict'])
    
    @staticmethod
    def freeze_model(module):
        for p in module.parameters():
            p.requires_grad = False

    def reconstruction_loss(self, h_origin, h_fake, log_variance):
        """L2 loss between original-regenerated features at cLSTM's last hidden layer"""
        return torch.norm(h_origin - h_fake, p=2)

    def prior_loss(self, mu, log_variance):
        """KL( q(e|x) || N(0,1) )"""
        return 0.5 * torch.sum(-1 + log_variance.exp() + mu.pow(2) - log_variance)

    def sparsity_loss(self, scores):
        """Summary-Length Regularization"""
        return torch.norm(torch.mean(scores) - self.config.summary_rate)

    def gan_loss(self, original_prob, fake_prob, uniform_prob):
        """Typical GAN loss + Classify uniformly scored features"""
        return torch.mean(torch.log(original_prob) + torch.log(1 - fake_prob) + torch.log(1 - uniform_prob))

    def contrastive_loss(self, positive_pairs, negative_pairs, margin=1.0, lambdac=1.0, adaptive_weighting=False):
        """Compute contrastive loss with optional adaptive weighting"""
        pos_distance = torch.norm(positive_pairs[0] - positive_pairs[1], p=2)
        neg_distance = torch.norm(negative_pairs[0] - negative_pairs[1], p=2)

        loss = (1 - torch.exp(-pos_distance)) + torch.relu(neg_distance - margin)

        if adaptive_weighting:
            total_distance = pos_distance + neg_distance
            adaptive_weight = pos_distance / (total_distance + 1e-8)
            loss *= adaptive_weight

        return lambdac * loss.mean()
    
    
    def train(self):
        step = 0
        for epoch_i in trange(self.config.n_epochs, desc='Epoch', ncols=80):
            s_e_loss_history = []
            d_loss_history = []
            c_loss_history = []
            
            for batch_i, (video_features, text_features,_) in enumerate(tqdm(self.train_loader, desc='Batch', ncols=80, leave=False)):
                if video_features.size(1) > 10000 or text_features.size(1) > 10000:
                    continue
                
                # Prepare features
                video_features = video_features.view(-1, self.config.video_input_size)
                text_features = text_features.view(-1, self.config.text_input_size)
                
                video_features = Variable(video_features).to(device=device)
                text_features = Variable(text_features).to(device=device)

                video_features = video_features.to(device=device)  # [B, N, 1024]
                text_features = text_features.to(device=device)    # [B, N, 768]

                # Train sLSTM, eLSTM
                if self.config.verbose:
                    tqdm.write('\nTraining sLSTM and eLSTM...')

                scores, h_mu, h_log_variance, generated_features,fused_video, fused_text = self.summarizer(
                    video_features, text_features)
                _, _, _, uniform_features,fused_video, fused_text = self.summarizer(
                    video_features, text_features, uniform=True)


                #fused_input = torch.cat([fused_video, fused_text], dim=1)  # [B, N_video + N_text, hidden_size]
                fused_input = fused_video  # [B, N_video, hidden_size], if we consider text-guided
                h_origin, original_prob = self.discriminator(fused_input)
 
                h_fake, fake_prob = self.discriminator(generated_features)
                h_uniform, uniform_prob = self.discriminator(uniform_features)

                # Calculate losses
                reconstruction_loss = self.reconstruction_loss(h_origin, h_fake, h_log_variance)
                prior_loss = self.prior_loss(h_mu, h_log_variance)
                sparsity_loss = self.sparsity_loss(scores)
        
                s_e_loss = reconstruction_loss + prior_loss + sparsity_loss 

                self.s_e_optimizer.zero_grad()
                s_e_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                self.s_e_optimizer.step()
                s_e_loss_history.append(s_e_loss.data)

                # Train dLSTM
                if self.config.verbose:
                    tqdm.write('Training dLSTM...')

                scores, h_mu, h_log_variance, generated_features,fused_video, fused_text = self.summarizer(
                    video_features, text_features)
                _, _, _, uniform_features,fused_video, fused_text = self.summarizer(
                    video_features, text_features, uniform=True)

                #fused_video, fused_text = self.fusion_module(video_features, text_features)
                #fused_input = torch.cat([fused_video, fused_text], dim=1)  # [B, N_video + N_text, hidden_size]
                fused_input = fused_video  # [B, N_video, hidden_size]
                h_origin, original_prob = self.discriminator(fused_input)
                #h_origin, original_prob = self.discriminator(torch.cat([video_features, text_features], dim=1))
                h_fake, fake_prob = self.discriminator(generated_features)
                h_uniform, uniform_prob = self.discriminator(uniform_features)

                reconstruction_loss = self.reconstruction_loss(h_origin, h_fake, h_log_variance)
                gan_loss = self.gan_loss(original_prob, fake_prob, uniform_prob)
                
                # Contrastive loss
                pos_pairs = (h_origin, h_fake)
                neg_pairs = (h_fake, h_uniform)
                contrastive_loss_value = self.contrastive_loss(pos_pairs, neg_pairs, adaptive_weighting=True)

                d_loss = reconstruction_loss + gan_loss + contrastive_loss_value

                self.d_optimizer.zero_grad()
                d_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                self.d_optimizer.step()
                d_loss_history.append(d_loss.data)

                # Train cLSTM
                if batch_i > self.config.discriminator_slow_start:
                    if self.config.verbose:
                        tqdm.write('Training cLSTM...')

                    scores, h_mu, h_log_variance, generated_features,fused_video, fused_text = self.summarizer(
                        video_features, text_features)
                    _, _, _, uniform_features,fused_video, fused_text = self.summarizer(
                        video_features, text_features, uniform=True)
                    
                    #fused_video, fused_text = self.fusion_module(video_features, text_features)
                    fused_input = torch.cat([fused_video, fused_text], dim=1)  # [B, N_video + N_text, hidden_size]
                    #fused_input = fused_video  # [B, N_video, hidden_size]
                    h_origin, original_prob = self.discriminator(fused_input)
                    #h_origin, original_prob = self.discriminator(torch.cat([video_features, text_features], dim=1))
                    h_fake, fake_prob = self.discriminator(generated_features)
                    h_uniform, uniform_prob = self.discriminator(uniform_features)

                    c_loss = -1 * self.gan_loss(original_prob, fake_prob, uniform_prob)
                    
                    pos_pairs = (h_origin, h_fake)
                    neg_pairs = (h_fake, h_uniform)
                    contrastive_loss_value = self.contrastive_loss(pos_pairs, neg_pairs, adaptive_weighting=True)
                    
                    c_loss += contrastive_loss_value

                    self.c_optimizer.zero_grad()
                    c_loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.clip)
                    self.c_optimizer.step()
                    c_loss_history.append(c_loss.data)

                # Logging
                if self.config.verbose:
                    self.writer.update_loss(reconstruction_loss.data, step, 'recon_loss')
                    self.writer.update_loss(prior_loss.data, step, 'prior_loss')
                    self.writer.update_loss(sparsity_loss.data, step, 'sparsity_loss')
                    self.writer.update_loss(gan_loss.data, step, 'gan_loss')
                    self.writer.update_loss(contrastive_loss_value.data, step, 'contrastive_loss')
                    #self.writer.update_loss(original_prob.data, step, 'original_prob')
                    self.writer.update_loss(original_prob.data.mean().item(), step, 'original_prob')
                    self.writer.update_loss(fake_prob.data, step, 'fake_prob')
                    self.writer.update_loss(uniform_prob.data, step, 'uniform_prob')
                

                step += 1

            # Epoch logging
            s_e_loss = torch.stack(s_e_loss_history).mean()
            d_loss = torch.stack(d_loss_history).mean()
            c_loss = torch.stack(c_loss_history).mean() if c_loss_history else torch.tensor(0.0)

            self.writer.update_loss(s_e_loss, epoch_i, 's_e_loss_epoch')
            self.writer.update_loss(d_loss, epoch_i, 'd_loss_epoch')
            self.writer.update_loss(c_loss, epoch_i, 'c_loss_epoch')

            # Save checkpoint
            if (self.n_epoch_trained + epoch_i) % 5 == 4:
                checkpoint_path = str(self.config.save_dir) + f'/epoch-{self.n_epoch_trained + epoch_i}.pkl'
                if not os.path.isdir(self.config.save_dir):
                    os.makedirs(self.config.save_dir)
                    
                torch.save({
                    'n_epoch_trained': self.n_epoch_trained + epoch_i + 1,
                    'model_state_dict': self.model.state_dict(),
                    'e_optimizer_state_dict': self.s_e_optimizer.state_dict(),
                    'd_optimizer_state_dict': self.d_optimizer.state_dict(),
                    'c_optimizer_state_dict': self.c_optimizer.state_dict(),
                }, checkpoint_path)

            self.evaluate(self.n_epoch_trained + epoch_i)
            self.model.train()

    def evaluate(self, epoch_i, video_name=None):
        print(f"\n=== Running evaluate() | Epoch: {epoch_i} | video_name: {video_name} ===")

        self.model.eval()
        out_dict = {}

        # Check if test_loader has data
        if len(self.test_loader) == 0:
            print("test_loader is empty.")
            return

        for video_features, text_features, video_data in tqdm(self.test_loader, desc='Evaluate', ncols=80, leave=False):
            print(" Evaluate loop")
            print(f" - video_features shape: {video_features.shape}")
            print(f" - text_features shape: {text_features.shape}")
            print(f" - video_data: {video_data}")

            video_features = video_features.view(-1, self.config.video_input_size).to(device)
            text_features = text_features.view(-1, self.config.text_input_size).to(device)

            with torch.no_grad():
                scores, h_mu, h_log_variance, decoded_features,fused_video, fused_text = self.summarizer(video_features, text_features)
                scores = scores.squeeze(1).cpu().numpy().tolist()
                print(f" Scores generated | Count: {len(scores)} | Sample: {scores[:5]}")

            if video_name:
                #out_dict[video_name] = scores
                video_key = "video_1"  # To make the video name consistent with the video name (video_1) in the .h5 file 
                out_dict[video_key] = scores  # of the test video during testing or inference 

            # Batch mode with video names
            elif isinstance(video_data, dict) and 'video_names' in video_data:
                video_names = video_data['video_names']
                if isinstance(video_names, (list, tuple)):
                    batch_size = len(video_names)
                    num_scores_per_video = len(scores) // batch_size
                    score_idx = 0
                    for name in video_names:
                        video_scores = scores[score_idx:score_idx + num_scores_per_video]
                        out_dict[name] = video_scores
                        score_idx += num_scores_per_video
                else:
                    out_dict[video_names] = scores
            else:
                out_dict[f"video_{len(out_dict)+1}"] = scores

            print(f"Current videos in out_dict: {list(out_dict.keys())}")

            # Save score
            score_save_path = self.config.score_dir.joinpath(f'{self.config.video_type}_{epoch_i}.json')
            print(f"\nSaving scores to: {score_save_path}")
            os.makedirs(self.config.score_dir, exist_ok=True)

            try:
                with open(score_save_path, 'w') as f:
                    tqdm.write(f'Saving score at {str(score_save_path)}.')
                    json.dump(out_dict, f, indent=4)
                os.chmod(score_save_path, 0o777)
                print("Score file saved successfully.")
            except Exception as e:
                print(f" Failed to save score: {e}")

        
        
    def pretrain(self):
        pass

if __name__ == '__main__':
    pass
