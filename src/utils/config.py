# src/utils/config.py
"""统一的配置读取助手。

仓库中的 YAML 配置只提交占位值（空字符串），真实的 API Key 通过环境变量
``OPENAI_API_KEY`` 注入，避免密钥被提交到 Git 历史中。

用法::

    from src.utils.config import resolve_api_key

    api_key = resolve_api_key(cfg.get("api_key"))
"""

import os

DEFAULT_ENV_VAR = "OPENAI_API_KEY"

# 出现这些占位值说明配置文件里还没有填真实的 Key
_PLACEHOLDERS = (
    "",
    "your_api_key",
    "your-api-key",
    "sk-xxx",
    "sk-...",
    "none",
    "null",
)


def resolve_api_key(api_key=None, env_var=DEFAULT_ENV_VAR, required=False):
    """优先使用显式传入的 Key，否则回退到环境变量。

    Args:
        api_key: 配置文件里读到的 Key，通常是占位值。
        env_var: 环境变量名，默认 ``OPENAI_API_KEY``。
        required: 为 True 时，拿不到 Key 直接抛异常。

    Returns:
        解析后的 Key 字符串；找不到且 ``required=False`` 时返回 ``None``。
    """
    if api_key and str(api_key).strip().lower() not in _PLACEHOLDERS:
        return str(api_key).strip()

    env_value = os.environ.get(env_var)
    if env_value and env_value.strip().lower() not in _PLACEHOLDERS:
        return env_value.strip()

    if required:
        raise RuntimeError(
            f"未找到 API Key。请在配置文件的 api_key 字段中填写，"
            f"或设置环境变量 {env_var}。"
        )
    return None
