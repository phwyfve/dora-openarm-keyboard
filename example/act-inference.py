"""Windows-native ACT adapter: MuJoCo observations -> trained joint targets."""

import argparse
from collections import deque
import time

import cv2
import dora
from huggingface_hub import snapshot_download
from lerobot.configs import PreTrainedConfig
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
import numpy as np
from PIL import Image
import pyarrow as pa
import torch


MODEL = "enactic/act-openarm-2-cell-pick_up_cube_mujoco"
REVISION = "27cbd9a205d2e006aef518a54a3c53cf8e421a13"
CAMERAS = ("ceiling", "head_left", "head_right", "wrist_left", "wrist_right")
QPOS_TYPE = pa.struct([("qpos", pa.list_(pa.float32()))])


def load_policy():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; install the CUDA PyTorch build first.")
    torch.set_num_threads(4)
    path = snapshot_download(
        MODEL, revision=REVISION, allow_patterns=["*.json", "*.safetensors"]
    )
    config = PreTrainedConfig.from_pretrained(path)
    config.pretrained_path = path
    config.device = "cuda"
    if config.input_features["observation.state"].shape != (16,):
        raise ValueError("This adapter requires a 16-joint observation.")
    if config.output_features["action"].shape != (16,):
        raise ValueError("This adapter requires 16-joint actions.")
    policy = get_policy_class(config.type).from_pretrained(
        config=config, pretrained_name_or_path=path
    )
    policy.eval()
    policy.reset()
    preprocessor, postprocessor = make_pre_post_processors(
        config,
        pretrained_path=path,
        preprocessor_overrides={"device_processor": {"device": "cuda"}},
    )
    print(f"[ACT] Loaded {MODEL}@{REVISION} on {torch.cuda.get_device_name(0)}", flush=True)
    return policy, preprocessor, postprocessor


@torch.inference_mode()
def predict(policy, preprocessor, postprocessor, batch):
    started = time.monotonic()
    chunk = policy.predict_action_chunk(preprocessor(batch))
    if chunk.ndim != 3 or chunk.shape[0] != 1 or chunk.shape[2] != 16:
        raise ValueError(f"Unexpected ACT output shape: {tuple(chunk.shape)}")
    count = chunk.shape[1]
    actions = postprocessor(chunk.reshape(count, 16)).cpu().numpy()
    if actions.shape != (count, 16) or not np.isfinite(actions).all():
        raise ValueError("ACT returned invalid joint targets.")
    print(
        f"[ACT] Predicted {count} actions in {time.monotonic() - started:.2f}s; "
        f"first right={np.round(actions[0, :8], 3).tolist()}",
        flush=True,
    )
    return actions


def image_tensor(encoded, shape):
    bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Cannot decode a MuJoCo camera JPEG.")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    _, height, width = shape
    if rgb.shape != (height, width, 3):
        rgb = np.asarray(Image.fromarray(rgb).resize((width, height)))
    return torch.from_numpy(rgb.copy()).permute(2, 0, 1).float() / 255.0


def run(node, policy, preprocessor, postprocessor):
    cameras = {}
    arms = {}
    tick = None
    actions = deque()
    expected_step = 0
    initial_qpos = None
    snapshot_timestamp = None
    for event in node:
        if event["type"] == "STOP":
            break
        if event["type"] == "ERROR":
            raise RuntimeError(event["error"])
        if event["type"] != "INPUT":
            continue
        metadata = event["metadata"]
        if metadata.get("control_step") != expected_step:
            raise ValueError(f"Unexpected snapshot: expected control step {expected_step}")
        timestamp = metadata["observation_timestamp"]
        if snapshot_timestamp is None:
            snapshot_timestamp = timestamp
        elif timestamp != snapshot_timestamp:
            raise ValueError("Images and arm states are not from the same snapshot")
        name = event["id"]
        if name.startswith("camera_"):
            camera = name.removeprefix("camera_")
            if camera not in CAMERAS or camera in cameras:
                raise ValueError(f"Unexpected or duplicate camera: {camera}")
            cameras[camera] = event["value"].to_numpy(zero_copy_only=False)
        elif name in ("state_right", "state_left"):
            if name in arms:
                raise ValueError(f"Duplicate arm state: {name}")
            qpos = np.asarray(event["value"][0]["qpos"].as_py(), dtype=np.float32)
            if qpos.shape != (8,) or not np.isfinite(qpos).all():
                raise ValueError(f"Invalid joint observation from {name}")
            arms[name] = qpos
        elif name == "tick" and tick is None:
            tick = metadata
        else:
            raise ValueError(f"Unexpected input: {name}")
        if len(cameras) != 5 or len(arms) != 2 or tick is None:
            continue
        actual = np.concatenate((arms["state_right"], arms["state_left"]))
        if initial_qpos is None:
            initial_qpos = actual.copy()
        if not actions:
            batch = {"observation.state": torch.from_numpy(actual)}
            for camera in CAMERAS:
                key = f"observation.images.{camera}"
                batch[key] = image_tensor(
                    cameras[camera], policy.config.input_features[key].shape
                )
            # Physics remains paused while this new chunk is predicted.
            actions.extend(predict(policy, preprocessor, postprocessor, batch))
        target = actions.popleft()
        for side, values in (("right", target[:8]), ("left", target[8:])):
            node.send_output(
                f"position_{side}",
                pa.array([{"qpos": values.tolist()}], type=QPOS_TYPE),
                metadata={"control_step": expected_step},
            )
        if expected_step % 100 == 0:
            print(
                f"[ACT] Executing action {expected_step + 1}; "
                f"simulated={tick['simulation_time']:.3f}s; "
                f"observed joint change={np.max(np.abs(actual - initial_qpos)):.3f} rad",
                flush=True,
            )
        expected_step += 1
        cameras.clear()
        arms.clear()
        tick = None
        snapshot_timestamp = None


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Load and warm up; no simulation.")
    args = parser.parse_args()
    node = None if args.check else dora.Node()
    policy, preprocessor, postprocessor = load_policy()
    if args.check:
        batch = {
            key: torch.zeros(feature.shape, dtype=torch.float32)
            for key, feature in policy.config.input_features.items()
        }
        result = predict(policy, preprocessor, postprocessor, batch)
        print(f"[ACT] Check passed: {result.shape}, finite={np.isfinite(result).all()}")
    else:
        run(node, policy, preprocessor, postprocessor)


if __name__ == "__main__":
    main()
