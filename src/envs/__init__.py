# src/envs/__init__.py
"""环境工厂。

本仓库只保留 AirSim 无人机环境；室内 Habitat 相关的环境实现已移除。
"""

from .airsim_env import AirSimEnv

__all__ = ["AirSimEnv", "construct_envs"]


def construct_envs(args):
    """按配置创建环境实例。目前只支持 ``environment: airsim``。"""
    environment = getattr(args, "environment", "airsim")
    if environment != "airsim":
        raise NotImplementedError(
            f"本仓库只保留 AirSim 环境，收到 environment='{environment}'。"
            "室内 Habitat 环境已不在维护范围内。"
        )
    return AirSimEnv(args)
