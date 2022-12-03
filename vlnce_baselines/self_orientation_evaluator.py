import json
import os
import time
import warnings
from collections import defaultdict
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import jsonlines
import torch
import tqdm
from habitat import logger
from habitat_baselines.common.baseline_registry import baseline_registry
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.common.tensorboard_utils import TensorboardWriter
from habitat_baselines.rl.ddppo.algo.ddp_utils import is_slurm_batch_job

from vlnce_baselines.common.env_utils import construct_envs_auto_reset_false
from vlnce_baselines.common.aux_losses import AuxLosses
from vlnce_baselines.common.base_il_trainer import BaseVLNCETrainer
from vlnce_baselines.common.recollection_dataset import (
    TeacherRecollectionDataset,
)


from habitat.utils.visualizations.utils import append_text_to_image
from habitat_baselines.common.environments import get_env_class
from habitat_baselines.common.obs_transformers import (
    apply_obs_transforms_batch,
)
from habitat_baselines.utils.common import batch_obs

from habitat_extensions.utils import generate_video, observations_to_image
from vlnce_baselines.common.utils import extract_instruction_tokens

from transformers import CLIPProcessor, CLIPModel
from PIL import Image

with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=FutureWarning)
    import tensorflow as tf  # noqa: F401

torch.set_printoptions(precision=2, sci_mode=False)

@baseline_registry.register_trainer(name="self_orientation_evaluator")
class SelfOrientationEvaluator(BaseVLNCETrainer):
    """A Teacher Forcing trainer that re-collects episodes from simulation
    rather than saving them all to disk. Included as starter code for the
    RxR-Habitat Challenge but can also train R2R agents.
    """

    supported_tasks: List[str] = ["VLN-v0"]

    def __init__(self, config=None):
        super().__init__(config)

    def _make_dirs(self) -> None:
        self._make_ckpt_dir()
        os.makedirs(
            os.path.dirname(
                self.config.IL.RECOLLECT_TRAINER.trajectories_file
            ),
            exist_ok=True,
        )
        if self.config.EVAL.SAVE_RESULTS:
            self._make_results_dir()

    def save_checkpoint(self, epoch: int, step_id: int) -> None:
        torch.save(
            obj={
                "state_dict": self.policy.state_dict(),
                "config": self.config,
                "optim_state": self.optimizer.state_dict(),
                "epoch": epoch,
                "step_id": step_id,
            },
            f=os.path.join(self.config.CHECKPOINT_FOLDER, f"ckpt.{epoch}.pth"),
        )
    
    def _eval_checkpoint(
        self,
        checkpoint_path: str,
        writer: TensorboardWriter,
        checkpoint_index: int = 0,
    ) -> None:
        """Evaluates a single checkpoint.

        Args:
            checkpoint_path: path of checkpoint
            writer: tensorboard writer object
            checkpoint_index: index of the current checkpoint
        """

        model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
        processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

        def self_orient(instruction: str, rgb_frames: list, num_rots: int = 12, rot_dir: int = 2):      
            views = []
            for _ in range(num_rots):
                outputs = envs.step([rot_dir])
                observations, _, dones, infos = [list(x) for x in zip(*outputs)]
                views.append(Image.fromarray(observations[0]['rgb'], mode='RGB'))
                # current_rgb_view = observations[0]['rgb']#.cpu().numpy()
                # instr = observations[0]['rxr_instruction']
                # print(instr)
                # print(observations)
                # plt.imshow(current_rgb_view)
                # plt.show()
                # plt.close()
                if dones[0]:
                    return observations, dones, infos, rgb_frames

            outputs = envs.step([rot_dir])
            
            # CLIP scores
            # import pdb; pdb.set_trace()
            inputs = processor(
                text=[instruction], images=views, return_tensors="pt", max_length=77)#padding=True, max_length=100)
            outputs = model(**inputs)
            logits_per_image = outputs.logits_per_image # this is the image-text similarity score
            probs = logits_per_image.softmax(dim=0)
            view_idx = np.argmax(probs.detach().numpy()) + 1
            
            for _ in range(view_idx):
                outputs = envs.step([rot_dir])
                observations, _, dones, infos = [list(x) for x in zip(*outputs)]
                
                if len(config.VIDEO_OPTION) > 0:
                    infos[i]["selforienting"] = True
                    frame = observations_to_image(observations[i], infos[i])
                    frame = append_text_to_image(
                        frame, current_episodes[i].instruction.instruction_text
                    )
                    rgb_frames[i].append(frame)
                
                if dones[0]:
                    return observations, dones, infos, rgb_frames

            return observations, dones, infos, rgb_frames

        logger.info(f"checkpoint_path: {checkpoint_path}")


        config = self.config.clone()
        if self.config.EVAL.USE_CKPT_CONFIG:
            ckpt = self.load_checkpoint(checkpoint_path, map_location="cpu")
            config = self._setup_eval_config(ckpt)

        split = config.EVAL.SPLIT

        config.defrost()
        config.TASK_CONFIG.DATASET.SPLIT = split
        config.TASK_CONFIG.DATASET.ROLES = ["guide"]
        config.TASK_CONFIG.DATASET.LANGUAGES = config.EVAL.LANGUAGES
        config.TASK_CONFIG.TASK.NDTW.SPLIT = split
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.SHUFFLE = False
        config.TASK_CONFIG.ENVIRONMENT.ITERATOR_OPTIONS.MAX_SCENE_REPEAT_STEPS = (
            -1
        )
        config.IL.ckpt_to_load = checkpoint_path
        config.use_pbar = not is_slurm_batch_job()

        # num_rotations = int(360 / config.SIMULATOR.TURN_ANGLE)
        num_rotations = int(360 / 30)

        if len(config.VIDEO_OPTION) > 0:
            config.TASK_CONFIG.TASK.MEASUREMENTS.append("TOP_DOWN_MAP_VLNCE")

        config.freeze()

        if config.EVAL.SAVE_RESULTS:
            fname = os.path.join(
                config.RESULTS_DIR,
                f"stats_ckpt_{checkpoint_index}_{split}.json",
            )
            if os.path.exists(fname):
                logger.info("skipping -- evaluation exists.")
                return

        envs = construct_envs_auto_reset_false(
            config, get_env_class(config.ENV_NAME)
        )
        observation_space, action_space = self._get_spaces(config, envs=envs)

        self._initialize_policy(
            config,
            load_from_ckpt=True,
            observation_space=observation_space,
            action_space=action_space,
        )
        self.policy.eval()

        observations = envs.reset()
        observations = extract_instruction_tokens(
            observations, self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID
        )
        batch = batch_obs(observations, self.device)
        batch = apply_obs_transforms_batch(batch, self.obs_transforms)

        rnn_states = torch.zeros(
            envs.num_envs,
            self.policy.net.num_recurrent_layers,
            config.MODEL.STATE_ENCODER.hidden_size,
            device=self.device,
        )
        prev_actions = torch.zeros(
            envs.num_envs, 1, device=self.device, dtype=torch.long
        )

        action_history = []

        not_done_masks = torch.zeros(
            envs.num_envs, 1, dtype=torch.uint8, device=self.device
        )

        stats_episodes = {}

        rgb_frames = [[] for _ in range(envs.num_envs)]
        is_self_orienting = [[] for _ in range(envs.num_envs)]
        if len(config.VIDEO_OPTION) > 0:
            os.makedirs(config.VIDEO_DIR, exist_ok=True)

        num_eps = sum(envs.number_of_episodes)
        if config.EVAL.EPISODE_COUNT > -1:
            num_eps = min(config.EVAL.EPISODE_COUNT, num_eps)

        pbar = tqdm.tqdm(total=num_eps) if config.use_pbar else None
        log_str = (
            f"[Ckpt: {checkpoint_index}]"
            " [Episodes evaluated: {evaluated}/{total}]"
            " [Time elapsed (s): {time}]"
        )
        start_time = time.time()

        entr_sum = 0.0
        entropy_history = []
        budget = 0
        while envs.num_envs > 0 and len(stats_episodes) < num_eps:
                
            current_episodes = envs.current_episodes()
            instruction = current_episodes[0].instruction.instruction_text

            if len(entropy_history) == self.config.IL.SELF_ORIENTATION.entropy_history and budget < self.config.IL.SELF_ORIENTATION.budget_per_episode:
                entr_sum = np.asarray(entropy_history).sum()

                entropy_history.pop(0)
                
                if entr_sum > self.config.IL.SELF_ORIENTATION.entropy_threshold:
                    # print(f"Need to self orient: {entr_sum}")
                    observations, dones, infos, rgb_frames = self_orient(
                        instruction=instruction, 
                        rgb_frames=rgb_frames, 
                        num_rots=num_rotations)
                    budget += 1
                    entropy_history = []
                    entr_sum = 0.0
                else:
                    with torch.no_grad():
                        actions, rnn_states, entr, prob = self.policy.act(
                            batch,
                            rnn_states,
                            prev_actions,
                            not_done_masks,
                            deterministic=not config.EVAL.SAMPLE,
                        )
                        prev_actions.copy_(actions)   
                    outputs = envs.step([a[0].item() for a in actions])
                    observations, _, dones, infos = [list(x) for x in zip(*outputs)]
            else:
                with torch.no_grad():
                    actions, rnn_states, entr, prob = self.policy.act(
                        batch,
                        rnn_states,
                        prev_actions,
                        not_done_masks,
                        deterministic=not config.EVAL.SAMPLE,
                    )
                    prev_actions.copy_(actions)
                outputs = envs.step([a[0].item() for a in actions])
                observations, _, dones, infos = [list(x) for x in zip(*outputs)]
                
            entropy_history.append(entr.item())

            # with torch.no_grad():
            #     actions, rnn_states, entr, probs = self.policy.act(
            #         batch,
            #         rnn_states,
            #         prev_actions,
            #         not_done_masks,
            #         deterministic=not config.EVAL.SAMPLE,
            #     )
            #     prev_actions.copy_(actions)

            # entropy_history.append(entr.item())
            # if len(entropy_history) == 10:
            #     # print(f"Entropy History: {np.asarray(entropy_history).sum()}")
            #     entr_sum = np.asarray(entropy_history).sum()

            #     entropy_history.pop(0)

            # outputs = envs.step([a[0].item() for a in actions])
            # observations, _, dones, infos = [list(x) for x in zip(*outputs)]

            # action_history.append(prev_actions[0][0].item())
            
            not_done_masks = torch.tensor(
                [[0] if done else [1] for done in dones],
                dtype=torch.uint8,
                device=self.device,
            )

            # reset envs and observations if necessary
            for i in range(envs.num_envs):
                if len(config.VIDEO_OPTION) > 0:
                    infos[i]["selforienting"] = False
                    frame = observations_to_image(observations[i], infos[i])
                    frame = append_text_to_image(
                        frame, current_episodes[i].instruction.instruction_text
                    )
                    rgb_frames[i].append(frame)

                if not dones[i]:
                    continue

                ep_id = current_episodes[i].episode_id
                stats_episodes[ep_id] = infos[i]
                observations[i] = envs.reset_at(i)[0]
                prev_actions[i] = torch.zeros(1, dtype=torch.long)
                budget = 0
                entropy_history = []
                entr_sum = 0.0

                if config.use_pbar:
                    pbar.update()
                else:
                    logger.info(
                        log_str.format(
                            evaluated=len(stats_episodes),
                            total=num_eps,
                            time=round(time.time() - start_time),
                        )
                    )

                if len(config.VIDEO_OPTION) > 0:
                    generate_video(
                        video_option=config.VIDEO_OPTION,
                        video_dir=config.VIDEO_DIR,
                        images=rgb_frames[i],
                        episode_id=ep_id,
                        checkpoint_idx=checkpoint_index,
                        metrics={"spl": stats_episodes[ep_id]["spl"]},
                        tb_writer=writer,
                    )
                    del stats_episodes[ep_id]["top_down_map_vlnce"]
                    rgb_frames[i] = []
                    

            observations = extract_instruction_tokens(
                observations,
                self.config.TASK_CONFIG.TASK.INSTRUCTION_SENSOR_UUID,
            )
            batch = batch_obs(observations, self.device)
            batch = apply_obs_transforms_batch(batch, self.obs_transforms)

            envs_to_pause = []
            next_episodes = envs.current_episodes()

            for i in range(envs.num_envs):
                if next_episodes[i].episode_id in stats_episodes:
                    envs_to_pause.append(i)

            (
                envs,
                rnn_states,
                not_done_masks,
                prev_actions,
                batch,
                rgb_frames,
            ) = self._pause_envs(
                envs_to_pause,
                envs,
                rnn_states,
                not_done_masks,
                prev_actions,
                batch,
                rgb_frames,
            )

        envs.close()
        if config.use_pbar:
            pbar.close()

        aggregated_stats = {}
        num_episodes = len(stats_episodes)
        for k in next(iter(stats_episodes.values())).keys():
            aggregated_stats[k] = (
                sum(v[k] for v in stats_episodes.values()) / num_episodes
            )

        if config.EVAL.SAVE_RESULTS:
            with open(fname, "w") as f:
                json.dump(aggregated_stats, f, indent=4)

        logger.info(f"Episodes evaluated: {num_episodes}")
        checkpoint_num = checkpoint_index + 1
        for k, v in aggregated_stats.items():
            logger.info(f"{k}: {v:.6f}")
            writer.add_scalar(f"eval_{split}_{k}", v, checkpoint_num)