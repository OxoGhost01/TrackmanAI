from pathlib import Path


def create_config_copy():
    base_dir = Path(__file__).resolve().parents[1]
    config_dir = base_dir / "config_files"

    config_copy_path = config_dir / "config_copy.py"
    if config_copy_path.exists():
        config_copy_path.unlink()

    config_text = (config_dir / "config.py").read_text(encoding="utf-8")
    user_config_text = (config_dir / "user_config.py").read_text(encoding="utf-8")
    input_list_text = (config_dir / "input_list.py").read_text(encoding="utf-8")

    combined = (
        config_text
        + "\n\n# user config\n"
        + user_config_text
        + "\n\n# input list\n"
        + input_list_text
    )

    config_copy_path.write_text(combined, encoding="utf-8")
