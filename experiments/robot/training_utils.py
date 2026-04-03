"""
finetune.py

Simple script for parameter-efficient fine-tuning of OpenVLA models loaded through the HuggingFace AutoClasses, using
HuggingFace PEFT library for low-rank adaptation (LoRA).

Notes & Benchmarks:
    - Requires PEFT (`pip install peft==0.11.1`)
    - LoRA fine-tuning (see parameters below -- no quantization, LoRA rank = 32, target_modules = all-linear):
        + One 48 GB GPU can fit a Batch Size of 12
        + One 80 GB GPU can fit a Batch Size of 24

Run with:
    - [Single Node Multi-GPU (= $K) ]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py
    - [Override Config Values]: torchrun --standalone --nnodes 1 --nproc-per-node $K vla-scripts/finetune.py \
                                    --data_root_dir <PATH/TO/RLDS/DATASETS/DIRECTORY> \
                                    --dataset_name <DATASET_NAME> \
                                    --run_root_dir <PATH/TO/LOGS/DIR> \
                                    ...
"""

import os
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
import numpy as np

import draccus
import torch
import torch.distributed as dist
import tqdm
from accelerate import PartialState
from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from torch.utils.data import DataLoader
from transformers import AutoModelForVision2Seq, AutoProcessor, BitsAndBytesConfig
from transformers import AutoConfig, AutoImageProcessor
from transformers.modeling_outputs import CausalLMOutputWithPast

import wandb
from prismatic.models.backbones.llm.prompting import PurePromptBuilder, VicunaV15ChatPromptBuilder
from prismatic.util.data_utils import PaddedCollatorForActionPrediction
from prismatic.vla.action_tokenizer import ActionTokenizer
from prismatic.vla.datasets import RLDSBatchTransform, RLDSDataset
from prismatic.vla.datasets.rlds.utils.data_utils import save_dataset_statistics

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

from experiments.robot.openvla_utils import get_processor, get_logprob_of_action
# Sane Defaults
os.environ["TOKENIZERS_PARALLELISM"] = "false"


# # === Utilities ===
# # fmt: off
# def create_vision_transform(vla: nn.Module, input_size: int) -> Callable[[Image.Image], torch.Tensor]:
#     """Gets image transform for the vision encoder."""
#     data_cfg = timm.data.resolve_model_data_config(vla.vision_backbone)
#     data_cfg["input_size"] = (3, input_size, input_size)
#     return timm.data.create_transform(
#         input_size=data_cfg["input_size"],
#         interpolation=data_cfg["interpolation"],
#         mean=data_cfg["mean"],
#         std=data_cfg["std"],
#         crop_pct=1.0,           # Set to 1.0 to disable cropping
#         crop_mode="center",     # Default crop mode --> no-op when `crop_pct == 1.0`
#         is_training=False,      # Disable image_aug when loading transform; handled by RLDS dataloader
#     )
#
# # fmt: on


@dataclass
class OnlineAdaptConfig:
    learning_rate: float = 5e-4
    use_lora: bool = True
    lora_rank: int = 32
    lora_dropout: float = 0.0
    center_crop: bool = True
    ppo_clip: float = 0.2
    reward_mode: str = "robomonkey"
    reward_server_port: int = 3100 

# @dataclass
# class RolloutEntry:
#     observation: dict
#     action_token_ids: torch.Tensor
#     reward: float
#     next_observation: dict | None
#     old_logprob: float | torch.Tensor

class OnlineAdapter:
    def __init__(self, model, processor, cfg: OnlineAdaptConfig | None = None):
        self.processor = processor
        self.cfg = cfg or OnlineAdaptConfig()
        self.model = self._attach_lora(model)
        self.optimizer = AdamW(
            [p for p in self.model.parameters() if p.requires_grad],
            lr=self.cfg.learning_rate,
        )

    def get_reward(self, obs, action_token_ids, task_description, img_path, redis_client=None, image_history=None):
        """ Gets reward for a given observation and action token IDs, using either the RoboMonkey reward model or a custom reward server. """
        if self.cfg.reward_mode == "robomonkey":
            from experiments.robot.robomonkey_utils import _get_rewards

            class _RewardCfg:
                reward_server_port = self.cfg.reward_server_port

            rewards = _get_rewards(task_description, img_path, action_token_ids, _RewardCfg)
            return float(rewards[0])
        elif self.cfg.reward_mode == "ttvla":
            if redis_client is None or image_history is None or len(image_history) < 2:
                return 0.0
            import json
            
            redis_client.rpush("tta_images", json.dumps({
                "obs" : image_history,
                "task_description": task_description,
            }))
            result_raw = redis_client.blpop("tta_rewards", timeout=30)
            if result_raw is None:
                print("Timeout while waiting for reward from TTVLA server.")
                return 0.0
            value_list = json.loads(result_raw[1])["value_list"]
            if len(value_list) < 2:
                return 0.0
            return float(value_list[-1]) - float(value_list[-2])
        else:
            raise ValueError(f"Invalid reward mode: {self.cfg.reward_mode}")


    def _attach_lora(self, model):
        lora_config = LoraConfig(
            r=self.cfg.lora_rank,
            lora_alpha=min(self.cfg.lora_rank, 16),
            lora_dropout=self.cfg.lora_dropout,
            target_modules="all-linear",
            init_lora_weights="gaussian",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
        return model

    def update(self, buffer: list, task_description: str, cfg):
        if not buffer:
            return {"num_updates": 0, "buffer_size": 0}

        self.model.train()
        self.optimizer.zero_grad()
        losses = []

        for observation, action, reward, obs_next, logp in buffer:
            new_logprob = get_logprob_of_action(
                cfg,
                self.model,
                observation,
                task_description,
                action,
                processor=self.processor,
                center_crop=self.cfg.center_crop,
            )
            old_logprob = torch.as_tensor(new_logprob).new_tensor(logp)
            reward_t = torch.as_tensor(new_logprob).new_tensor(reward)

            ratio = torch.exp(new_logprob - old_logprob)
            clipped = torch.clamp(ratio, 1.0 - self.cfg.ppo_clip, 1.0 + self.cfg.ppo_clip)
            losses.append(-torch.minimum(ratio * reward_t, clipped * reward_t))

        loss = torch.stack(losses).mean()
        loss.backward()
        self.optimizer.step()
        self.model.eval()

        return {"num_updates": 1, "buffer_size": len(buffer), "loss": loss.item()}





# @dataclass
# class FinetuneConfig:
#     # fmt: off
#     vla_path: str = "openvla/openvla-7b"                            # Path to OpenVLA model (on HuggingFace Hub)

#     # Directory Paths
#     data_root_dir: Path = Path("datasets/open-x-embodiment")        # Path to Open-X dataset directory
#     dataset_name: str = "droid_wipe"                                # Name of fine-tuning dataset (e.g., `droid_wipe`)
#     run_root_dir: Path = Path("runs")                               # Path to directory to store logs & checkpoints
#     adapter_tmp_dir: Path = Path("adapter-tmp")                     # Temporary directory for LoRA weights before fusing

#     # Fine-tuning Parameters
#     batch_size: int = 16                                            # Fine-tuning batch size
#     max_steps: int = 200_000                                        # Max number of fine-tuning steps
#     save_steps: int = 5000                                          # Interval for checkpoint saving
#     learning_rate: float = 5e-4                                     # Fine-tuning learning rate
#     grad_accumulation_steps: int = 1                                # Gradient accumulation steps
#     image_aug: bool = True                                          # Whether to train with image augmentations
#     shuffle_buffer_size: int = 100_000                              # Dataloader shuffle buffer size (can reduce if OOM)
#     save_latest_checkpoint_only: bool = True                        # Whether to save only one checkpoint per run and
#                                                                     #   continually overwrite the latest checkpoint
#                                                                     #   (If False, saves all checkpoints)

#     # LoRA Arguments
#     use_lora: bool = True                                           # Whether to use LoRA fine-tuning
#     lora_rank: int = 32                                             # Rank of LoRA weight matrix
#     lora_dropout: float = 0.0                                       # Dropout applied to LoRA weights
#     use_quantization: bool = False                                  # Whether to 4-bit quantize VLA for LoRA fine-tuning
#                                                                     #   => CAUTION: Reduces memory but hurts performance

#     # Tracking Parameters
#     wandb_project: str = "openvla"                                  # Name of W&B project to log to (use default!)
#     wandb_entity: str = "stanford-voltron"                          # Name of entity to log under
#     run_id_note: Optional[str] = None                               # Extra note for logging, Weights & Biases


#     tta_type: str = "ttvla"
#     center_crop: bool = True 
#     # fmt: on


# def finetune(vla, processor, buffer, task_description, cfg: FinetuneConfig=FinetuneConfig() ) -> None:
#     print(f"Fine-tuning OpenVLA Model `{cfg.vla_path}` with `{cfg.tta_type}`")

#     # [Validate] Ensure GPU Available & Set Device / Distributed Context
#     assert torch.cuda.is_available(), "Fine-tuning assumes at least one GPU is available!"
#     # distributed_state = PartialState()
#     # torch.cuda.set_device(device_id := distributed_state.local_process_index)
#     # torch.cuda.empty_cache()

#     # Configure Unique Experiment ID & Log Directory
#     exp_id = (
#         f"{cfg.vla_path.split('/')[-1]}+{cfg.dataset_name}"
#         f"+b{cfg.batch_size * cfg.grad_accumulation_steps}"
#         f"+lr-{cfg.learning_rate}"
#     )
#     if cfg.use_lora:
#         exp_id += f"+lora-r{cfg.lora_rank}+dropout-{cfg.lora_dropout}"
#     if cfg.use_quantization:
#         exp_id += "+q-4bit"
#     if cfg.run_id_note is not None:
#         exp_id += f"--{cfg.run_id_note}"
#     if cfg.image_aug:
#         exp_id += "--image_aug"

#     # Start =>> Build Directories
#     # run_dir, adapter_dir = cfg.run_root_dir / exp_id, cfg.adapter_tmp_dir / exp_id
#     # os.makedirs(run_dir, exist_ok=True)

#     # Quantization Config =>> only if LoRA fine-tuning
#     quantization_config = None
#     if cfg.use_quantization:
#         assert cfg.use_lora, "Quantized training only supported for LoRA fine-tuning!"
#         quantization_config = BitsAndBytesConfig(
#             load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_quant_type="nf4"
#         )

#     # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
#     # AutoConfig.register("openvla", OpenVLAConfig)
#     # AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
#     # AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
#     # AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    
#     # Device Placement =>> note that BitsAndBytes automatically handles for quantized training
#     # if cfg.use_quantization:
#     #     vla = prepare_model_for_kbit_training(vla)
#     # else:
#     #     vla = vla.to(device_id)

#     # [LoRA] Wrap Model w/ PEFT `LoraConfig` =>> by default we set `target_modules=all-linear`
#     if cfg.use_lora:
#         lora_config = LoraConfig(
#             r=cfg.lora_rank,
#             lora_alpha=min(cfg.lora_rank, 16),
#             lora_dropout=cfg.lora_dropout,
#             target_modules="all-linear",
#             init_lora_weights="gaussian",
#         )
#         vla = get_peft_model(vla, lora_config)
#         vla.print_trainable_parameters()

#     # Wrap VLA in PyTorch DDP Wrapper for Multi-GPU Training
#     # vla = DDP(vla, device_ids=[device_id], find_unused_parameters=True, gradient_as_bucket_view=True)

#     # Create Optimizer =>> note that we default to a simple constant learning rate!
#     trainable_params = [param for param in vla.parameters() if param.requires_grad]
#     optimizer = AdamW(trainable_params, lr=cfg.learning_rate)

#     # Create Action Tokenizer
#     action_tokenizer = ActionTokenizer(processor.tokenizer)

#     # Load Fine-tuning Dataset =>> note that we use an RLDS-formatted dataset following Open X-Embodiment by default.
#     #   =>> If you want to use a non-RLDS dataset (e.g., a standard PyTorch Dataset) see the following commented block.
#     #   =>> Note that our training code does not loop over epochs because the RLDS loader does this implicitly; if using
#     #       your own Dataset, make sure to add the appropriate logic to the training loop!
#     #
#     # ---
#     # from prismatic.vla.datasets import DummyDataset
#     #
#     # vla_dataset = DummyDataset(
#     #     action_tokenizer,
#     #     processor.tokenizer,
#     #     image_transform=processor.image_processor.apply_transform,
#     #     prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
#     # )
#     # ---
#     # batch_transform = RLDSBatchTransform(
#     #     action_tokenizer,
#     #     processor.tokenizer,
#     #     image_transform=processor.image_processor.apply_transform,
#     #     prompt_builder_fn=PurePromptBuilder if "v01" not in cfg.vla_path else VicunaV15ChatPromptBuilder,
#     # )
#     # vla_dataset = RLDSDataset(
#     #     cfg.data_root_dir,
#     #     cfg.dataset_name,
#     #     batch_transform,
#     #     resize_resolution=tuple(vla.module.config.image_sizes),
#     #     shuffle_buffer_size=cfg.shuffle_buffer_size,
#     #     image_aug=cfg.image_aug,
#     # )

#     # # [Important] Save Dataset Statistics =>> used to de-normalize actions for inference!
#     # if distributed_state.is_main_process:
#     #     save_dataset_statistics(vla_dataset.dataset_statistics, run_dir)

#     # Create Collator and DataLoader
#     # collator = PaddedCollatorForActionPrediction(
#     #     processor.tokenizer.model_max_length, processor.tokenizer.pad_token_id, padding_side="right"
#     # )
#     # dataloader = DataLoader(
#     #     vla_dataset,
#     #     batch_size=cfg.batch_size,
#     #     sampler=None,
#     #     collate_fn=collator,
#     #     num_workers=0,  # Important =>> Set to 0 if using RLDS; TFDS rolls its own parallelism!
#     # )

#     # Initialize Logging =>> W&B
#     # if distributed_state.is_main_process:
#     #     wandb.init(entity=cfg.wandb_entity, project=cfg.wandb_project, name=f"ft+{exp_id}")

#     # Deque to store recent train metrics (used for computing smoothened metrics for gradient accumulation)
#     # recent_losses = deque(maxlen=cfg.grad_accumulation_steps)
#     # recent_action_accuracies = deque(maxlen=cfg.grad_accumulation_steps)
#     # recent_l1_losses = deque(maxlen=cfg.grad_accumulation_steps)

#     # Train!
    
#     vla.train()
#     optimizer.zero_grad()
        
#     for idx, (observation, action, reward, obs_next, logp) in enumerate(buffer):
#         logp_new= get_logprob_of_action(
#             cfg,
#             vla, 
#             observation,
#             task_description,
#             action,
#             processor=processor,
#             center_crop=cfg.center_crop,
#         )

#         ratio = torch.exp(logp_new - logp)
#         loss = -torch.minimum(ratio * reward, torch.clamp(ratio, 0.8, 1.2) * reward)
            

#         # Backward pass
#         loss.backward()

            


            
#         optimizer.step()
#         optimizer.zero_grad()
                    

    
   