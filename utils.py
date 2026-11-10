import yaml
import os

RESULTS_BASE = "/opt/scratch/marta/results"

def load_yaml(file_path):
    with open(file_path, 'r') as file:
        return yaml.safe_load(file)

def save_yaml(data, file_path):
    with open(file_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

def deep_merge(dict1, dict2):
    for key, value in dict2.items():
        if key in dict1 and isinstance(dict1[key], dict) and isinstance(value, dict):
            deep_merge(dict1[key], value)
        else:
            dict1[key] = value
    return dict1

def merge_dicts(*dicts):
    if len(dicts) == 0:
        return {}
    if len(dicts) == 1:
        return dicts[0]

    result = dicts[0]
    for d in dicts[1:]:
        result = deep_merge(result, d)
    return result

def is_oom_message(msg: str) -> bool:
    msg = msg.lower()
    return "out of memory" in msg or ("cuda" in msg and "alloc" in msg)

PERMANENT_ERROR_MARKERS = ("AssertionError", "SyntaxError", "TypeError",
                            "ImportError", "ModuleNotFoundError", "KeyError",
                            "AttributeError", "NameError")

def is_permanent_failure(err_msg) -> bool:
    """True if a crash message indicates a code error rather than an OOM."""
    if not err_msg:
        return False  # unknown cause (e.g. legacy checkpoint) → assume retriable
    return any(marker in err_msg for marker in PERMANENT_ERROR_MARKERS)