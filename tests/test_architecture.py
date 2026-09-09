from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_lerobot_is_an_external_dependency() -> None:
    assert not (ROOT / "src" / "lerobot").exists()
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert '"lerobot[feetech,intelrealsense,smolvla]==0.4.4"' in pyproject


def test_hand_tracking_streamer_stays_a_submodule_not_a_python_dependency() -> None:
    gitmodules = (ROOT / ".gitmodules").read_text(encoding="utf-8")
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8").lower()
    assert "scripts/hand-tracking-streamer" in gitmodules
    assert "hand-tracking-streamer" not in pyproject


def test_source_does_not_import_private_sodexarm_modules_from_lerobot() -> None:
    invalid = (
        "lerobot.model.so101_kinematics",
        "lerobot.robots.so101_amazinghand",
        "lerobot.teleoperators.quest_hts",
    )
    for path in (ROOT / "src" / "sodexarm").rglob("*.py"):
        source = path.read_text(encoding="utf-8")
        assert not any(name in source for name in invalid), path
