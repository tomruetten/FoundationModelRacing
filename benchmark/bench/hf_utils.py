"""Checkpoint helpers that need no GPU stack (testable on a laptop)."""

from pathlib import Path


def resolve_repo_relative_vlm(config: dict, checkpoint: str, *, file_exists=None, snapshot_download=None) -> str:
    """Make a repository-relative `vlm_name_or_path` loadable and return the path to load the model from.

    `config` is the raw config.json dict (from `PretrainedConfig.get_config_dict`) and is patched in place. It must
    be fixed *before* the config object is built: the Alpamayo config constructor already loads a processor from
    `vlm_name_or_path`.

    Some checkpoints (e.g. SIMFORGE-D1.5) ship the Qwen3-VL config and tokenizer in a `vlm/` sub-folder and set
    `vlm_name_or_path: "vlm"` in config.json. That only works from a local checkout; given a Hub id transformers
    looks for a Hub model called "vlm". In that case the snapshot is downloaded into the normal Hugging Face
    cache (same data volume as a regular from_pretrained) and `config['vlm_name_or_path']` is pointed at the local
    folder. Checkpoints with a Hub-id or absolute path are returned unchanged and download nothing extra.
    """
    vlm = config.get("vlm_name_or_path")
    if not vlm or Path(vlm).is_absolute() or Path(vlm).exists():
        return checkpoint

    local = Path(checkpoint)
    if local.is_dir():
        if (local / vlm).is_dir():
            config["vlm_name_or_path"] = str(local / vlm)
        return checkpoint

    if file_exists is None or snapshot_download is None:
        from huggingface_hub import file_exists as _fe, snapshot_download as _sd

        file_exists, snapshot_download = file_exists or _fe, snapshot_download or _sd
    if not file_exists(checkpoint, f"{vlm}/config.json"):
        return checkpoint  # a real Hub id such as "Qwen/Qwen3-VL-8B-Instruct"
    snapshot = snapshot_download(checkpoint, ignore_patterns=["benchmark/*"])
    config["vlm_name_or_path"] = str(Path(snapshot) / vlm)
    return snapshot
