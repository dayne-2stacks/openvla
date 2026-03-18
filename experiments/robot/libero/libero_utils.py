"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os
from pathlib import Path

import imageio
import numpy as np
import tensorflow as tf
from experiments.robot.robot_utils import (
    DATE,
    DATE_TIME,
)
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from libero.libero.envs.textures import get_texture_file_list
from libero.libero.envs.utils import postprocess_model_xml
import xml.etree.ElementTree as ET

SUPPORTED_SHIFT_NAMES = {"none", "appearance"}
SUPPORTED_SHIFT_MODES = {"noise", "blur", "gamma", "texture"}
SEVERITY_TO_GAMMA_OFFSET = [0.05, 0.10, 0.15, 0.20, 0.25]
SEVERITY_TO_NOISE_STD = [3.0, 6.0, 9.0, 12.0, 15.0]
SEVERITY_TO_BLUR_SIGMA = [0.4, 0.8, 1.2, 1.6, 2.0]
MAX_BLUR_KERNEL_SIZE = 13
WALL_GEOM_NAMES = {
    "wall_leftcorner_visual",
    "wall_rightcorner_visual",
    "wall_left_visual",
    "wall_right_visual",
    "wall_rear_visual",
    "wall_front_visual",
}



def _to_hw_tuple(resize_size):
    """Converts image resize size to (height, width)."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        return (resize_size, resize_size)
    assert len(resize_size) == 2
    return resize_size


def _episode_seed_sequence(seed, task_id, episode_idx):
    """Creates a deterministic SeedSequence from (seed, task_id, episode_idx)."""
    return np.random.SeedSequence([int(seed), int(task_id), int(episode_idx)])


def _episode_seed(seed, task_id, episode_idx):
    """Creates a deterministic seed from (seed, task_id, episode_idx)."""
    seed_sequence = _episode_seed_sequence(seed, task_id, episode_idx)
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


def _get_gaussian_kernel_2d(sigma, max_kernel_size=MAX_BLUR_KERNEL_SIZE):
    """Returns a normalized 2D Gaussian kernel."""
    if sigma <= 0:
        return None
    radius = int(math.ceil(3.0 * sigma))
    kernel_size = 2 * radius + 1
    if kernel_size > max_kernel_size:
        kernel_size = max_kernel_size
        if kernel_size % 2 == 0:
            kernel_size -= 1
        radius = kernel_size // 2
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel_1d = np.exp(-(coords**2) / (2.0 * (sigma**2)))
    kernel_1d = kernel_1d / np.sum(kernel_1d)
    kernel_2d = np.outer(kernel_1d, kernel_1d)
    kernel_2d = kernel_2d / np.sum(kernel_2d)
    return kernel_2d.astype(np.float32)


def _apply_gaussian_blur(img, sigma):
    """Applies Gaussian blur using TensorFlow depthwise convolution."""
    if sigma <= 0:
        return img
    kernel_2d = _get_gaussian_kernel_2d(sigma)
    if kernel_2d is None:
        return img
    channels = img.shape[-1]
    depthwise_kernel = np.repeat(kernel_2d[:, :, None, None], channels, axis=2)
    img_tf = tf.convert_to_tensor(img[None, ...], dtype=tf.float32)
    kernel_tf = tf.convert_to_tensor(depthwise_kernel, dtype=tf.float32)
    img_tf = tf.nn.depthwise_conv2d(img_tf, kernel_tf, strides=[1, 1, 1, 1], padding="SAME")
    return img_tf[0].numpy()


def build_episode_shift_state(cfg, resize_size, task_id, episode_idx):
    """Builds deterministic shift state for one (task, episode)."""
    resize_size = _to_hw_tuple(resize_size)
    if cfg.shift_name == "none":
        return {
            "enabled": False,
            "task_id": task_id,
            "episode_idx": episode_idx,
            "seed": None,
            "gamma": None,
            "noise_std": None,
            "blur_sigma": None,
            "noise_map": None,
        }

    if cfg.shift_name != "appearance":
        raise ValueError(f"Unsupported shift_name: {cfg.shift_name}")
    if cfg.shift_mode not in SUPPORTED_SHIFT_MODES:
        raise ValueError(f"Unsupported shift_mode: {cfg.shift_mode}")
    if not isinstance(cfg.severity, int) or not (1 <= cfg.severity <= 5):
        raise ValueError(f"Expected severity to be an integer in [1, 5], got: {cfg.severity}")

    severity_idx = cfg.severity - 1
    gamma_offset = SEVERITY_TO_GAMMA_OFFSET[severity_idx]
    noise_std = SEVERITY_TO_NOISE_STD[severity_idx]
    blur_sigma = SEVERITY_TO_BLUR_SIGMA[severity_idx]

    seed_sequence = _episode_seed_sequence(cfg.seed, task_id, episode_idx)
    seed = _episode_seed(cfg.seed, task_id, episode_idx)
    rng = np.random.default_rng(seed_sequence)
    gamma_sign = -1.0 if rng.random() < 0.5 else 1.0
    gamma = 1.0 + gamma_sign * gamma_offset
    noise_map = None
    if cfg.shift_mode == "noise":
        noise_map = rng.normal(loc=0.0, scale=noise_std, size=(resize_size[0], resize_size[1], 3)).astype(np.float32)

    return {
        "enabled": True,
        "task_id": task_id,
        "episode_idx": episode_idx,
        "seed": seed,
        # Always keep scalar params populated so logging stays robust across single-mode runs.
        "gamma": gamma,
        "noise_std": noise_std,
        "blur_sigma": blur_sigma,
        "noise_map": noise_map,
    }


def apply_shift(img, cfg, shift_state):
    """Applies a configured shift to a preprocessed image."""
    if cfg.shift_name == "none" or not shift_state["enabled"]:
        return img
    if cfg.shift_name != "appearance":
        raise ValueError(f"Unsupported shift_name: {cfg.shift_name}")
    if cfg.shift_mode not in SUPPORTED_SHIFT_MODES:
        raise ValueError(f"Unsupported shift_mode: {cfg.shift_mode}")
    if cfg.shift_mode == "texture":
        # Texture shift is applied in the simulator at reset-time (not in image postprocessing).
        return img

    shifted_img = img.astype(np.float32)

    if cfg.shift_mode == "gamma":
        # Gamma correction (lighting proxy)
        shifted_img = 255.0 * np.power(np.clip(shifted_img / 255.0, 0.0, 1.0), shift_state["gamma"])

    elif cfg.shift_mode == "noise":
        # Additive Gaussian sensor noise
        shifted_img = shifted_img + shift_state["noise_map"]

    elif cfg.shift_mode == "blur":
        # Defocus/motion blur proxy
        shifted_img = _apply_gaussian_blur(shifted_img, shift_state["blur_sigma"])

    shifted_img = np.clip(shifted_img, 0.0, 255.0).astype(np.uint8)
    return shifted_img


def get_libero_env(task, model_family, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution}
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description



def get_xml(env):
    if not hasattr(env, "_texture_shift_base_xml"):
        xml = env.sim.model.get_xml()
        xml = postprocess_model_xml(xml, {})
        env._texture_shift_base_xml = xml
    xml = env._texture_shift_base_xml

    return xml


def replace_target_textures(env, severity, seed):

    if not isinstance(severity, int) or not (1 <= severity <= 4):
        raise ValueError(f"Expected severity to be an integer in [1,4], got: {severity}")

    rng = np.random.default_rng(seed)

    xml = get_xml(env)
    root = ET.fromstring(xml)

    asset = root.find("asset")
    worldbody = root.find("worldbody")

    if asset is None or worldbody is None:
        return {"applied": False, "swapped_material_count": 0, "target_material_count": 0}

    textures = {t.get("name"): t for t in asset.findall("texture") if t.get("name")}
    materials = {m.get("name"): m for m in asset.findall("material") if m.get("name")}
    assets_path = get_libero_path("assets")
    wall_texture_files = [
        texture_file
        for _, texture_file in get_texture_file_list(type="wall", texture_path=assets_path)
        if os.path.exists(texture_file)
    ]
    floor_texture_files = [
        texture_file
        for _, texture_file in get_texture_file_list(type="floor", texture_path=assets_path)
        if os.path.exists(texture_file)
    ]
    table_texture_files = [
        texture_file
        for _, texture_file in get_texture_file_list(type="table", texture_path=assets_path)
        if os.path.exists(texture_file)
    ]
    table_pool = list(dict.fromkeys(floor_texture_files + table_texture_files))

    wall_texture_file = None
    table_texture_file = None

    if severity in (2, 4) and wall_texture_files:
        wall_texture_file = str(rng.choice(wall_texture_files))

    if severity in (3, 4) and table_pool:
        table_texture_file = str(rng.choice(table_pool))

    target_materials = {
        "wall": set(),
        "table": set(),
    }

    # --------------------------------------------------
    # Traverse worldbody once
    # --------------------------------------------------

    for elem in worldbody.iter():

        if elem.tag not in {"geom", "body"}:
            continue

        name = elem.get("name")
        if name is None:
            continue

        material_name = elem.get("material")
        if material_name not in materials:
            continue

        # Allow explicit wall/table geom names to be targeted even if the current material has no texture yet.
        if name.lower() in WALL_GEOM_NAMES:
            group = "wall"
            target_materials[group].add(material_name)
            continue

        texture_name = materials[material_name].get("texture")
        if texture_name not in textures:
            continue

        if textures[texture_name].get("file") is None:
            continue

        searchable_name = " ".join(
            [
                name.lower(),
                material_name.lower(),
                texture_name.lower(),
                textures[texture_name].get("file", "").lower(),
            ]
        )
        if any(k in searchable_name for k in {"wall", "plaster", "sky", "background"}):
            group = "wall"
        elif any(k in searchable_name for k in {"floor", "ceramic", "table"}):
            group = "table"
        else:
            continue

        target_materials[group].add(material_name)

    swapped = []

    # --------------------------------------------------
    # Swap wall textures
    # --------------------------------------------------

    if wall_texture_file is not None:
        for m in target_materials["wall"]:
            texture_name = materials[m].get("texture")
            if texture_name not in textures:
                continue
            if textures[texture_name].get("file") is None:
                continue
            textures[texture_name].set("file", wall_texture_file)
            swapped.append(m)

    # --------------------------------------------------
    # Swap table textures
    # --------------------------------------------------

    if table_texture_file is not None:
        for m in target_materials["table"]:
            texture_name = materials[m].get("texture")
            if texture_name not in textures:
                continue
            if textures[texture_name].get("file") is None:
                continue
            textures[texture_name].set("file", table_texture_file)
            swapped.append(m)

    if not swapped:
        return {
            "applied": False,
            "swapped_material_count": 0,
            "target_material_count": 0,
        }

    updated_xml = ET.tostring(root, encoding="utf8").decode("utf8")

    env.reset_from_xml_string(updated_xml)

    return {
        "applied": True,
        "swapped_material_count": len(swapped),
        "target_material_count": sum(len(v) for v in target_materials.values()),
        "swapped_materials": sorted(swapped),
    }

def get_libero_dummy_action(model_family: str):
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def resize_image(img, resize_size):
    """
    Takes numpy array corresponding to a single image and returns resized image as numpy array.

    NOTE (Moo Jin): To make input images in distribution with respect to the inputs seen at training time, we follow
                    the same resizing scheme used in the Octo dataloader, which OpenVLA uses for training.
    """
    assert isinstance(resize_size, tuple)
    # Resize to image size expected by model
    img = tf.image.encode_jpeg(img)  # Encode as JPEG, as done in RLDS dataset builder
    img = tf.io.decode_image(img, expand_animations=False, dtype=tf.uint8)  # Immediately decode back
    img = tf.image.resize(img, resize_size, method="lanczos3", antialias=True)
    img = tf.cast(tf.clip_by_value(tf.round(img), 0, 255), tf.uint8)
    img = img.numpy()
    return img


def get_libero_image(obs, resize_size):
    """Extracts image from observations and preprocesses it."""
    assert isinstance(resize_size, int) or isinstance(resize_size, tuple)
    if isinstance(resize_size, int):
        resize_size = (resize_size, resize_size)
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    img = resize_image(img, resize_size)
    return img


def save_rollout_video(rollout_images, idx, success, task_description,shift=None, log_file=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{DATE}"
    if shift is not None:
        rollout_dir = f"./rollouts/{DATE}/{shift}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--episode={idx}--success={success}--task={processed_task_description}.mp4"
    video_writer = imageio.get_writer(mp4_path, fps=30)
    for img in rollout_images:
        video_writer.append_data(img)
    video_writer.close()
    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path

def save_image(img, task_id, episode_idx, t, task_description, shift=None, log_file=None):
    """Saves an image."""
    image_dir = f"./images/{DATE}"
    if shift is not None:
        image_dir = f"./images/{DATE}/{shift}"
    os.makedirs(image_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    image_path = f"{image_dir}/{DATE_TIME}--image={task_id}--episode={episode_idx}--t={t}--task={processed_task_description}.jpg"
    imageio.imwrite(image_path, img)
    print(f"Saved image at path {image_path}")
    if log_file is not None:
        log_file.write(f"Saved image at path {image_path}\n")

    abs_path = str(Path(image_path).resolve())
    
    return abs_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
