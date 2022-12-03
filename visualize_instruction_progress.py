#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import json 

import numpy as np

import habitat
from habitat.core.utils import try_cv2_import
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower
from habitat.utils.visualizations import maps
from habitat.utils.visualizations.utils import images_to_video

cv2 = try_cv2_import()
from vlnce_baselines.common.environments import VLNCEInferenceEnv
from vlnce_baselines.config.default import get_config

from habitat_extensions.utils import generate_video, observations_to_image, append_text_to_image

IMAGE_DIR = os.path.join("examples", "images")
if not os.path.exists(IMAGE_DIR):
    os.makedirs(IMAGE_DIR)

def instruction_progess():
    # config = habitat.get_config(config_paths="habitat_extensions/config/rxr_vlnce_english_task.yaml")
    config = get_config(config_paths="vlnce_baselines/config/rxr_baselines/rxr_cma_en.yaml")

    val_seen_data = 'data/datasets/RxR_VLNCE_v0/val_seen'
    vs_guide_gt = json.load(open(f'{val_seen_data}/val_seen_guide_gt.json'))
    
    with VLNCEInferenceEnv(config=config) as env:      
        print("Environment creation successful")
        
        # import pdb; pdb.set_trace()
        dirname = os.path.join(IMAGE_DIR, f"instruction_progress/")
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        os.makedirs(dirname)
        
        for episode in range(20):
            env.reset()

            current_episode = env.habitat_env.current_episode
            episode_id = current_episode.episode_id
            instruction = current_episode.instruction.instruction_text
            timed_instruction = current_episode.instruction.timed_instruction
            
            
            # some tokens dont have time stamp
            for t in range(len(timed_instruction)):
                if 'start_time' in timed_instruction[t]:
                    start_time = timed_instruction[t]['start_time']
                    break 
            for t in reversed(range(len(timed_instruction))):
                if 'end_time' in timed_instruction[t]:
                    end_time = timed_instruction[t]['end_time']
                    break 

            print(f"Start time {start_time} End time: {end_time}")
            instruction_time = end_time - start_time
            
            action_path = vs_guide_gt[episode_id]['actions']
            step_percentage = 1.0 / len(action_path)
            
            prev_path_progress = 0
            new_path_progress = 0

            frames = []
            action = 0
            while not env.habitat_env.episode_over:
                token_mask  = np.zeros(shape=(len(timed_instruction)))
                
                next_action = action_path[action]
                if next_action == 0:
                    break
                action += 1

                new_path_progress += step_percentage

                observations, reward, done, info = env.step(next_action)

                for i, token in enumerate(timed_instruction):
                    # print(token)
                    if 'end_time' not in token:
                        token_mask[i] = 1
                        continue
                    
                    token_time_percentage = token['end_time'] / instruction_time
                    if token_time_percentage > new_path_progress:
                        break

                    if token_time_percentage >= prev_path_progress:
                        token_mask[i] = 1
                
                # visualize 
                frame = observations_to_image(observations, info)
                frame = append_text_to_image(frame, timed_instruction, token_mask)
                frames.append(frame)                
                # import matplotlib.pyplot as plt
                # plt.imshow(frame)
                # plt.show()

                # update progress 
                prev_path_progress += step_percentage

            video_name = f"episode={episode_id}"
            images_to_video(frames, dirname, video_name, fps=1)
                
if __name__ == "__main__":
    instruction_progess()
