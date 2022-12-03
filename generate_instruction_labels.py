#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import os
import shutil
import json 
import re

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
    print()
    with VLNCEInferenceEnv(config=config) as env:      
        print("Environment creation successful")
        
        # import pdb; pdb.set_trace()
        dirname = os.path.join(IMAGE_DIR, f"instruction_progress/")
        if os.path.exists(dirname):
            shutil.rmtree(dirname)
        os.makedirs(dirname)
        
        for episode in range(20):
            print("*"*20)
            env.reset()

            current_episode = env.habitat_env.current_episode
            episode_id = current_episode.episode_id

            # 1. Split sentences into sub-instructions (by punctuation)
            instruction = current_episode.instruction.instruction_text
            split_instruction = [s.strip() for s in re.split(r'[,.!?]', instruction) if len(s) > 1 ]
            # split_instruction = [s for s in re.findall( r'\w+|[^\s\w]+', instruction) if len(s) > 1]
            
            # 2. Aggregate token tmings into sentence timings
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

            num_actions = len(vs_guide_gt[episode_id]['actions'])
            num_locations = len(vs_guide_gt[episode_id]['locations'])
            total_instruction_time = end_time - start_time

            time_for_each_action = total_instruction_time / num_locations
            # breakpoint()
            # Move forward, take a left after the sofa! and then stop
            # {Move} {forward,} {take} {a} ... {sofa! and then stop}
            # phrase = ""
            # phrase = "Move"
            # phrase = "Move foward,"
            # all_phrases = ["Move foward,"], phrase = ""
            # phrase = "take"
            # phrase = "take a"
            # phrase = "take a left"
            # .
            # .
            # phrase = "take a left after the sofa! and then stop"
            # all_phrases = ["Move foward,", "take a left after the sofa!", "and then stop"]
            all_phrases_with_times = []
            curr_phrase = ""
            curr_phrase_start_time = 0
            curr_phrase_end_time = 0
            for i, sub_phrase in enumerate(timed_instruction):
                # print(sub_phrase)
                word, st, et = sub_phrase['word'], sub_phrase.get('start_time', -1), sub_phrase.get('end_time', -1)
                old_st = st
                old_et = et
                if st == -1:
                    if i == 0:
                        st = 0
                    else:
                        st = timed_instruction[i-1]['end_time']
                        timed_instruction[i]['start_time'] = st
                if et == -1:
                    if i==len(timed_instruction)-1:
                        break
                    else:
                        et = timed_instruction[i+1]['start_time']
                        timed_instruction[i]['end_time'] = et
                # if old_et == -1 or old_st == -1:
                #     print(f"{timed_instruction[i-1]['word']}, {timed_instruction[i-1]['start_time']}, {timed_instruction[i-1]['end_time']}")
                #     print(f"{word}, {st}, {et}")
                if curr_phrase == "":
                    curr_phrase_start_time = st
                curr_phrase += f"{word.strip()} "
                print(curr_phrase)
                match = re.search(r"[.,!?]", curr_phrase)
                if match:
                    print("match", match)
                    breakpoint()

                # temp_str += f" {word}"
                # print(temp_str, phrase)
                # if word not in phrase:
                    #     print(word, phrase)
                    # sub_instr['word'] = re.sub(r'[,.!?]', '', sub_instr)
            
                
                





                

            
            
            # action_path = vs_guide_gt[episode_id]['actions']
            # step_percentage = 1.0 / len(action_path)
            
            # prev_path_progress = 0
            # new_path_progress = 0

            # frames = []
            # action = 0
            # while not env.habitat_env.episode_over:
            #     token_mask  = np.zeros(shape=(len(timed_instruction)))
                
            #     next_action = action_path[action]
            #     if next_action == 0:
            #         break
            #     action += 1

            #     new_path_progress += step_percentage

            #     observations, reward, done, info = env.step(next_action)

            #     for i, token in enumerate(timed_instruction):
            #         # print(token)
            #         if 'end_time' not in token:
            #             token_mask[i] = 1
            #             continue
                    
            #         token_time_percentage = token['end_time'] / instruction_time
            #         if token_time_percentage > new_path_progress:
            #             break

            #         if token_time_percentage >= prev_path_progress:
            #             token_mask[i] = 1
                
            #     # visualize 
            #     frame = observations_to_image(observations, info)
            #     frame = append_text_to_image(frame, timed_instruction, token_mask)
            #     frames.append(frame)                
            #     # import matplotlib.pyplot as plt
            #     # plt.imshow(frame)
            #     # plt.show()

            #     # update progress 
            #     prev_path_progress += step_percentage

            # video_name = f"episode={episode_id}"
            # images_to_video(frames, dirname, video_name, fps=1)
                
if __name__ == "__main__":
    instruction_progess()
