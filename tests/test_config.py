from codeguard_ai.config import load_config


def test_default_config_uses_project_local_paths() -> None:
    config = load_config()
    assert config.paths.root.name == "CodeGuardAI"
    assert config.paths.artifacts == config.paths.root / "artifacts" / "cpu_v3"
    assert load_config(quick=True).paths.artifacts == config.paths.artifacts / "smoke"
    assert config.project.model_name == "microsoft/codebert-base"
    assert config.training.max_length == 400


def test_cpu_v3_head_settings() -> None:
    config = load_config()
    assert config.head.hidden_sizes == [512, 128]
    assert config.head.epochs == 24
    assert config.head.patience == 5
    # Quick mode keeps a separate epoch budget and never shares the research namespace.
    quick = load_config(quick=True)
    assert quick.head.epochs == quick.quick_test.epochs
    assert quick.max_length == quick.quick_test.max_length
