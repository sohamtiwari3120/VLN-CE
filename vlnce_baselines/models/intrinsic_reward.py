import os
import numpy as np
import torch
import time
from sklearn.neighbors import NearestNeighbors
from torchmetrics.functional import pairwise_cosine_similarity
from transformers import CLIPProcessor, CLIPModel


class IntrinsicReward:

    def __init__(self,combine='sum'):
        #TODO: Put models and inputs on GPU
        assert combine in ['sum','product']
        self.combine = combine
        self.device = torch.device('cuda:0')
        self.embedding_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(self.device)
        self.processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
        self.novel_states = []
        self.num_neighbors = 5
        self.max_novel_states = 999
        self.knn_curiosity_tree = None
        self.sample_size = 50 
        self.reward_scale = 0.25
        self.curiosity_reward_scale = 1.0
        self.alignment_reward_scale = 1.0
        self.time_penalty = 0.0
        self.start_time = time.time() - 900
        self.num_hours = 0
        self.save_dir = '/home/ubuntu/VLN-CE/train_output'
        self.trajectory_alignment_scores = []
        self.curiosity_scores = []

    def factorize_obs(self,observations):
        #TODO: Populate this method
        obs_per_env = []
        for obs in observations:
            rgb_obs = []
            text = obs['instruction']['text']
            for index in range(0,12):
                string_index = '_' + str(index) if index > 0 else ''
                rgb_obs.append(obs['rgb' + string_index])
            obs_per_env.append((text,rgb_obs))
        return obs_per_env

    
    def log_outputs(self):
        if time.time() - self.start_time > 900:
            self.start_time = time.time()
            self.num_hours += 1
            novel_state_path = os.path.join(self.save_dir,'novel_states_' + str(self.num_hours))
            alignment_score_path = os.path.join(self.save_dir,'alignment_score_' + str(self.num_hours))
            curiosity_score_path =  os.path.join(self.save_dir,'curiosity_score_' + str(self.num_hours))
            np.save(novel_state_path,self.novel_states)
            np.save(alignment_score_path,self.trajectory_alignment_scores)
            np.save(curiosity_score_path,self.curiosity_scores)
            self.trajectory_alignment_scores = []
            self.curiosity_scores = []
        

    def compute_reward(self,observations): 
        try:
            obs_per_env = self.factorize_obs(observations)
            rewards = []
            for obs in obs_per_env:
                text,rgb_obs = obs
                reward = self.reward_per_env(text,rgb_obs) - self.time_penalty
                rewards.append(self.reward_scale*reward)
            self.log_outputs()
        except:
            rewards = np.zeros(len(observations))
            print('-------- ERROR THROWN IN INTRINSIC_REWARD ---------')
        
        return rewards

    def state_curiosity(self,rgb_embeddings):
        if len(self.novel_states) == 0:
            self.add_novel_embeddings(rgb_embeddings)
        nn_distances,_ = self.knn_curiosity_tree.kneighbors(rgb_embeddings.cpu().detach().numpy(),n_neighbors=self.num_neighbors,return_distance=True)
        rewards = np.sum(nn_distances,axis=1)
        self.add_novel_embeddings(rgb_embeddings) 
        return rewards*self.curiosity_reward_scale

    def sigmoid(self,x):
        return 1 / (1 + np.exp(-x))
            
    def add_novel_embeddings(self,rgb_embeddings):
        for rgb_embedding in rgb_embeddings:
            if len(self.novel_states) >= self.max_novel_states:
                #sampled_indices = np.random.choice(len(self.novel_states),self.sample_size,replace=False)
                sampled_embeddings = np.array(self.novel_states) #[sampled_indices] 
                nn_distances,_ = self.knn_curiosity_tree.kneighbors(sampled_embeddings,n_neighbors=self.num_neighbors+1,return_distance=True)
                min_distance = np.min(np.sum(nn_distances,axis=1))
                min_index = np.argmin(np.sum(nn_distances,axis=1))
                nn_distances,_ = self.knn_curiosity_tree.kneighbors([rgb_embedding.cpu().detach().numpy()],n_neighbors=self.num_neighbors,return_distance=True)
                if np.sum(nn_distances) > min_distance:
                    self.novel_states[min_index] = rgb_embedding.cpu().detach().numpy()
            else:
                self.novel_states.append(rgb_embedding.cpu().detach().numpy())

        self.knn_curiosity_tree = NearestNeighbors().fit(self.novel_states)


    def reward_per_env(self,text,rgb_obs):
        instruction_list = text.split('.')       
        inputs = self.processor(text = instruction_list, images=rgb_obs,return_tensors='pt',padding=True)
        rgb_embeddings = self.embedding_model.get_image_features(inputs['pixel_values'].to(self.device))
        text_embeddings = self.embedding_model.get_text_features(inputs['input_ids'].to(self.device),inputs['attention_mask'].to(self.device))
        reward = self.trajectory_alignment(rgb_embeddings,text_embeddings)
        self.trajectory_alignment_scores.append(reward)
        if len(self.trajectory_alignment_scores) > 1:
            reward = 5*(reward - np.min(self.trajectory_alignment_scores))/(np.max(self.trajectory_alignment_scores) - np.min(self.trajectory_alignment_scores))
        reward = self.sigmoid(reward) - 1.0
        if self.combine == 'sum':
            curiosity_reward = np.sum(self.state_curiosity(rgb_embeddings))
            self.curiosity_scores.append(curiosity_reward)
            if len(self.curiosity_scores) > 1:
                curiosity_reward = 5*(curiosity_reward - np.min(self.curiosity_scores))/(np.max(self.curiosity_scores) - np.min(self.curiosity_scores))
            curiosity_reward = self.sigmoid(curiosity_reward) - 1
            reward += curiosity_reward
        return reward
          

    def trajectory_alignment(self,rgb_embeddings,text_embeddings):
        similarity_matrix = pairwise_cosine_similarity(text_embeddings,rgb_embeddings)
        similarity_per_frame,instruction_indices = torch.max(similarity_matrix,dim=0)
        if self.combine == 'product':
            curiosity_value = self.state_curiosity(rgb_embeddings)
            similarity_per_frame = similarity_per_frame*curiosity_value
        #Later instructions should be rewarded more since it indicates that the embodied agent has progressed further 
        #similarity_per_frame = similarity_per_frame*(2**instruction_indices)
        similarity_per_frame,_ = torch.topk(similarity_per_frame,k=3) 
        return np.sum(similarity_per_frame.cpu().detach().numpy())*self.alignment_reward_scale



if __name__ == '__main__':
    pass   

        

        
        

            






