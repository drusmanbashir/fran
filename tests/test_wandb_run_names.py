from fran.managers import wandb as wandb_mod
from utilz.random_word_maker import (
    logical_word_capacity,
    ordered_word_suffixes,
    pattern_capacity,
)


def test_logical_word_capacity_matches_requested_scheme():
    assert logical_word_capacity() == 145530


def test_ordered_suffixes_progress_from_three_to_four_letters():
    suffixes = ordered_word_suffixes()
    first_three = [next(suffixes) for _ in range(3)]
    assert first_three == ["bab", "bac", "bad"]

    remaining_three_letter = pattern_capacity("CVC") - 3
    for _ in range(remaining_three_letter):
        last_three = next(suffixes)

    assert last_three == "zuz"
    assert next(suffixes) == "baba"


def test_digit_suffixes_follow_plain_word_spaces():
    suffixes = ordered_word_suffixes()
    total_plain = pattern_capacity("CVC") + pattern_capacity("CVCV")
    for _ in range(total_plain):
        last_plain = next(suffixes)

    assert last_plain == "zuzu"
    assert next(suffixes) == "bab0"


def test_new_run_id_skips_existing_project_suffixes(monkeypatch):
    class _Run:
        def __init__(self, run_id=None, name=None):
            self.id = run_id
            self.name = name

    class _Api:
        def runs(self, _path):
            return [_Run("KITS-bab", None), _Run("legacy-id", "KITS-bac")]

    monkeypatch.setattr(wandb_mod.wandb, "Api", lambda: _Api())
    monkeypatch.delenv("WANDB_MODE", raising=False)

    assert wandb_mod._new_run_id("entity", "kits", width=5) == "KITS-bad"
