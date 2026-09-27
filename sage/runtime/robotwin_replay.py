"""Report observed replay agreement without calling JPEGs lossless snapshots."""
import cv2
import h5py
import numpy as np


def inspect_handoff(path, index, joint_command, views, camera_datasets):
    if index < 1:
        raise ValueError("Tail handoff must have a preceding action block")
    with h5py.File(path, "r") as handle:
        expected = np.asarray(handle["lewm/action_blocks"][index - 1, 0], np.float32)
        observed = np.asarray(joint_command, np.float32)
        if expected.shape != observed.shape or not np.array_equal(expected, observed):
            raise RuntimeError("RoboTwin replay joint-command mismatch at handoff")
        images = {}
        for name, dataset in camera_datasets.items():
            value = handle[dataset][index]
            raw = value.tobytes() if isinstance(value, np.ndarray) else bytes(value)
            decoded = cv2.imdecode(np.frombuffer(raw, np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError(f"Invalid reference JPEG: {dataset}[{index}]")
            reference = cv2.cvtColor(decoded, cv2.COLOR_BGR2RGB)
            actual = np.asarray(views[name])
            if actual.shape != reference.shape:
                raise RuntimeError(f"Replay image shape mismatch: {name}")
            delta = np.abs(actual.astype(np.float32) - reference.astype(np.float32))
            images[name] = {"mae_255": float(delta.mean()), "max_abs_255": float(delta.max())}
    return {"capture_index": index, "joint_command_exact": True,
            "image_differences": images, "render_certificate_passed": False,
            "note": "JPEG/render differences are diagnostic, not an exact-physics certificate"}
