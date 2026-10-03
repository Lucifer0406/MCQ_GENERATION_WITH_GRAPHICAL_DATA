import pytest

from mcq import config


@pytest.fixture
def clean_env(monkeypatch):
    """Remove API-key vars for one test. setenv first so monkeypatch restores the original state,
    including undoing anything load_dotenv writes into os.environ."""
    for var in config.API_KEY_VARS:
        monkeypatch.setenv(var, "placeholder")
        monkeypatch.delenv(var)
    return monkeypatch


def test_missing_key_gives_actionable_error(clean_env, tmp_path):
    clean_env.setattr(config, "ENV_FILE", tmp_path / "missing.env")
    with pytest.raises(RuntimeError, match="GOOGLE_API_KEY"):
        config.get_api_key()


def test_key_read_from_env_file(clean_env, tmp_path):
    env = tmp_path / ".env"
    env.write_text('GOOGLE_API_KEY = "abc123"\n')  # same spacing/quotes style as the real .env
    clean_env.setattr(config, "ENV_FILE", env)
    assert config.get_api_key() == "abc123"


def test_paths_are_absolute_and_rooted_at_project():
    assert config.PROJECT_ROOT.is_absolute()
    assert (config.PROJECT_ROOT / "mcq" / "config.py").exists()
    assert config.RAW_DIR.parent == config.KNOWLEDGE_DIR
