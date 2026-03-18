"""Utils for evaluating the OpenVLA policy."""

import json
import os
import time

import numpy as np
import tensorflow as tf
import torch
from PIL import Image
from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.extern.hf.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor

# Initialize important constants and pretty-printing mode in NumPy.
ACTION_DIM = 7
DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")
DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
np.set_printoptions(formatter={"float": lambda x: "{0:0.3f}".format(x)})

# Initialize system prompt for OpenVLA v0.1.
OPENVLA_V01_SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)


def get_vla(cfg):
    """Loads and returns a VLA model from checkpoint."""
    # Load VLA checkpoint.
    print("[*] Instantiating Pretrained VLA model")
    print("[*] Loading in BF16 with Flash-Attention Enabled")

    # Register OpenVLA model to HF Auto Classes (not needed if the model is on HF Hub)
    AutoConfig.register("openvla", OpenVLAConfig)
    AutoImageProcessor.register(OpenVLAConfig, PrismaticImageProcessor)
    AutoProcessor.register(OpenVLAConfig, PrismaticProcessor)
    AutoModelForVision2Seq.register(OpenVLAConfig, OpenVLAForActionPrediction)

    vla = AutoModelForVision2Seq.from_pretrained(
        cfg.pretrained_checkpoint,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        load_in_8bit=cfg.load_in_8bit,
        load_in_4bit=cfg.load_in_4bit,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    )

    # Move model to device.
    # Note: `.to()` is not supported for 8-bit or 4-bit bitsandbytes models, but the model will
    #       already be set to the right devices and casted to the correct dtype upon loading.
    if not cfg.load_in_8bit and not cfg.load_in_4bit:
        vla = vla.to(DEVICE)

    # Load dataset stats used during finetuning (for action un-normalization).
    dataset_statistics_path = os.path.join(cfg.pretrained_checkpoint, "dataset_statistics.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        vla.norm_stats = norm_stats
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )

    return vla


def get_processor(cfg):
    """Get VLA model's Hugging Face processor."""
    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint, trust_remote_code=True)
    return processor


def crop_and_resize(image, crop_scale, batch_size):
    """
    Center-crops an image to have area `crop_scale` * (original image area), and then resizes back
    to original size. We use the same logic seen in the `dlimp` RLDS datasets wrapper to avoid
    distribution shift at test time.

    Args:
        image: TF Tensor of shape (batch_size, H, W, C) or (H, W, C) and datatype tf.float32 with
               values between [0,1].
        crop_scale: The area of the center crop with respect to the original image.
        batch_size: Batch size.
    """
    # Convert from 3D Tensor (H, W, C) to 4D Tensor (batch_size, H, W, C)
    assert image.shape.ndims == 3 or image.shape.ndims == 4
    expanded_dims = False
    if image.shape.ndims == 3:
        image = tf.expand_dims(image, axis=0)
        expanded_dims = True

    # Get height and width of crop
    new_heights = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))
    new_widths = tf.reshape(tf.clip_by_value(tf.sqrt(crop_scale), 0, 1), shape=(batch_size,))

    # Get bounding box representing crop
    height_offsets = (1 - new_heights) / 2
    width_offsets = (1 - new_widths) / 2
    bounding_boxes = tf.stack(
        [
            height_offsets,
            width_offsets,
            height_offsets + new_heights,
            width_offsets + new_widths,
        ],
        axis=1,
    )

    # Crop and then resize back up
    image = tf.image.crop_and_resize(image, bounding_boxes, tf.range(batch_size), (224, 224))

    # Convert back to 3D Tensor (H, W, C)
    if expanded_dims:
        image = image[0]

    return image


def get_vla_action(
    vla,
    processor,
    base_vla_name,
    obs,
    task_label,
    unnorm_key,
    center_crop=False,
    return_probs=False,
    do_sample=False,
    temperature=1.0,
):
    """Generate an action with the VLA policy, optionally with token logprobs."""
    image = Image.fromarray(obs["full_image"]).convert("RGB")

    if center_crop:
        batch_size = 1
        crop_scale = 0.9
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = crop_and_resize(image, crop_scale, batch_size)
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)
        image = Image.fromarray(image.numpy()).convert("RGB")

    if "openvla-v01" in base_vla_name:
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: "
            f"What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

    input_ids = inputs["input_ids"]
    if not torch.all(input_ids[:, -1] == 29871):
        input_ids = torch.cat(
            (input_ids, torch.tensor([[29871]], device=input_ids.device, dtype=input_ids.dtype)),
            dim=1,
        )

    gen_kwargs = dict(
        input_ids=input_ids,
        pixel_values=inputs["pixel_values"],
        max_new_tokens=vla.get_action_dim(unnorm_key),
        do_sample=do_sample,
    )
    if do_sample:
        gen_kwargs["temperature"] = temperature

    if return_probs:
        gen_kwargs["output_scores"] = True
        gen_kwargs["return_dict_in_generate"] = True

    gen = vla.generate(**gen_kwargs)
    sequences = gen.sequences if return_probs else gen

    action_token_ids = sequences[0, -vla.get_action_dim(unnorm_key):]
    action_token_ids_np = action_token_ids.detach().cpu().numpy()

    discretized_actions = vla.vocab_size - action_token_ids_np
    discretized_actions = np.clip(discretized_actions - 1, a_min=0, a_max=vla.bin_centers.shape[0] - 1)
    normalized_actions = vla.bin_centers[discretized_actions]

    action_norm_stats = vla.get_action_stats(unnorm_key)
    mask = action_norm_stats.get("mask", np.ones_like(action_norm_stats["q01"], dtype=bool))
    action_high = np.array(action_norm_stats["q99"])
    action_low = np.array(action_norm_stats["q01"])
    actions = np.where(
        mask,
        0.5 * (normalized_actions + 1) * (action_high - action_low) + action_low,
        normalized_actions,
    )

    if not return_probs:
        return actions

    step_log_probs = []
    for step, tok_id in enumerate(action_token_ids):
        step_scores = gen.scores[step][0].float()
        step_log_probs.append(torch.log_softmax(step_scores, dim=-1)[tok_id])

    step_log_probs = torch.stack(step_log_probs)
    sequence_log_prob = step_log_probs.sum()

    return {
        "action": actions,
        "action_token_ids": action_token_ids_np,
        "step_log_probs": step_log_probs.detach().cpu().numpy(),
        "sequence_log_prob": sequence_log_prob.item(),
    }


def get_logprob_of_action(
    cfg,
    model,
    obs,
    task_label,
    action_token_ids,
    processor=None,
    center_crop=False,
    temperature=1.0,
):
    """
    Returns log pi_theta(action_token_ids | obs, task_label).

    `action_token_ids` should be the sampled tokenized action from rollout time,
    shape [action_dim] or [1, action_dim].
    """
    image = Image.fromarray(obs["full_image"]).convert("RGB")

    if center_crop:
        batch_size = 1
        crop_scale = 0.9
        image = tf.convert_to_tensor(np.array(image))
        orig_dtype = image.dtype
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = crop_and_resize(image, crop_scale, batch_size)
        image = tf.clip_by_value(image, 0, 1)
        image = tf.image.convert_image_dtype(image, orig_dtype, saturate=True)
        image = Image.fromarray(image.numpy()).convert("RGB")

    if "openvla-v01" in cfg.pretrained_checkpoint:
        prompt = (
            f"{OPENVLA_V01_SYSTEM_PROMPT} USER: "
            f"What action should the robot take to {task_label.lower()}? ASSISTANT:"
        )
    else:
        prompt = f"In: What action should the robot take to {task_label.lower()}?\nOut:"

    inputs = processor(prompt, image).to(DEVICE, dtype=torch.bfloat16)

    prompt_ids = inputs["input_ids"]
    prompt_mask = inputs["attention_mask"]

    # Match predict_action() prompt formatting
    if not torch.all(prompt_ids[:, -1] == 29871):
        extra = torch.tensor([[29871]], device=prompt_ids.device, dtype=prompt_ids.dtype)
        prompt_ids = torch.cat([prompt_ids, extra], dim=1)
        prompt_mask = torch.cat(
            [prompt_mask, torch.ones((1, 1), device=prompt_mask.device, dtype=prompt_mask.dtype)],
            dim=1,
        )

    if not torch.is_tensor(action_token_ids):
        action_token_ids = torch.tensor(action_token_ids, device=prompt_ids.device, dtype=prompt_ids.dtype)
    else:
        action_token_ids = action_token_ids.to(device=prompt_ids.device, dtype=prompt_ids.dtype)

    if action_token_ids.ndim == 1:
        action_token_ids = action_token_ids.unsqueeze(0)

    full_input_ids = torch.cat([prompt_ids, action_token_ids], dim=1)
    full_attention_mask = torch.cat(
        [
            prompt_mask,
            torch.ones(
                (prompt_mask.shape[0], action_token_ids.shape[1]),
                device=prompt_mask.device,
                dtype=prompt_mask.dtype,
            ),
        ],
        dim=1,
    )

    outputs = model(
        input_ids=full_input_ids,
        attention_mask=full_attention_mask,
        pixel_values=inputs["pixel_values"],
        output_projector_features=True,
        return_dict=True,
    )

    logits = outputs.logits[0].float()
    n_patches = outputs.projector_features.shape[1]
    prompt_len = prompt_ids.shape[1]

    step_log_probs = []
    for k in range(action_token_ids.shape[1]):
        tok_id = action_token_ids[0, k]

        # Logit position that predicts token k in the action suffix.
        logit_idx = n_patches + prompt_len - 1 + k

        step_logits = logits[logit_idx]
        if temperature != 1.0:
            step_logits = step_logits / temperature

        step_log_probs.append(torch.log_softmax(step_logits, dim=-1)[tok_id])

    step_log_probs = torch.stack(step_log_probs)
    sequence_log_prob = step_log_probs.sum()

    return sequence_log_prob

