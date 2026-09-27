"""RoboTwin runtime initialization used by the paper's original evaluator."""
import os

import sapien.core as sapien
import yaml


class Sapien_TEST:
    """Retain the renderer probe until process exit (SAPIEN beta requirement)."""
    def __init__(self):
        from sapien.render import set_global_config
        set_global_config(max_num_materials=50000, max_num_textures=50000)
        self.engine = sapien.Engine()
        self.renderer = sapien.SapienRenderer()
        self.engine.set_renderer(self.renderer)
        sapien.render.set_camera_shader_dir("rt")
        sapien.render.set_ray_tracing_samples_per_pixel(32)
        sapien.render.set_ray_tracing_path_depth(8)
        sapien.render.set_ray_tracing_denoiser("oidn")
        self.scene = self.engine.create_scene(sapien.SceneConfig())


def embodiment_args(args):
    from envs._GLOBAL_CONFIGS import CONFIGS_PATH
    with open(os.path.join(CONFIGS_PATH, "_embodiment_config.yml"), encoding="utf-8") as f:
        embodiments = yaml.safe_load(f)

    def load_config(kind):
        path = embodiments[kind]["file_path"]
        with open(os.path.join(path, "config.yml"), encoding="utf-8") as f:
            return path, yaml.safe_load(f)

    kind = args["embodiment"]
    if len(kind) != 1:
        raise ValueError("The paper's A2B task uses a single dual-arm embodiment")
    path, config = load_config(kind[0])
    args.update(left_robot_file=path, right_robot_file=path,
                left_embodiment_config=config, right_embodiment_config=config,
                dual_arm_embodied=True, embodiment_name=str(kind[0]))
