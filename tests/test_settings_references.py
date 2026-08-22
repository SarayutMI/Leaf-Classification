# tests/test_settings_references.py
"""Guard against typos in `settings.X` lookups.

The rest of the suite mocks the model layer, so a bad attribute inside
load_models() only surfaces when the container boots. This check reads the
source instead of running it, and catches the whole class of mistake.
"""
import ast
import pathlib

from src.config import Settings

REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent


def _settings_attribute_refs():
    for directory in ("src", "scripts"):
        for path in (REPO_ROOT / directory).rglob("*.py"):
            if path.name == "config.py":
                continue
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Attribute)
                    and isinstance(node.value, ast.Name)
                    and node.value.id == "settings"
                ):
                    yield path.relative_to(REPO_ROOT), node.lineno, node.attr


def test_every_settings_reference_is_a_declared_field():
    fields = set(Settings.model_fields)
    unknown = [
        f"{path}:{lineno}: settings.{attr}"
        for path, lineno, attr in _settings_attribute_refs()
        if attr not in fields
    ]
    assert not unknown, "settings attributes that do not exist:\n" + "\n".join(unknown)


def test_config_module_is_not_shadowed_by_the_rename():
    """`tf.config`, `os.path` and friends must survive the config -> settings
    rename — `tf.settings.threading` is what broke the first container boot."""
    for directory in ("src", "scripts"):
        for path in (REPO_ROOT / directory).rglob("*.py"):
            tree = ast.parse(path.read_text())
            for node in ast.walk(tree):
                if (
                    isinstance(node, ast.Attribute)
                    and node.attr == "settings"
                    and isinstance(node.value, ast.Name)
                ):
                    raise AssertionError(
                        f"{path.relative_to(REPO_ROOT)}:{node.lineno}: "
                        f"`{node.value.id}.settings` is almost certainly a bad "
                        f"rename of `{node.value.id}.config`"
                    )
