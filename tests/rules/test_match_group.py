# tests/rules/test_match_group.py
"""Ranking rules for the decision tree. Rules are passed in explicitly, so
these never touch the database."""
from src.rules.service import match_group


def _rule(code, name="กลุ่ม", shape="Cordate", apex="Acute",
          base="Auriculate", margin="Entire"):
    return {"code": code, "name": name, "shape": shape, "apex": apex,
            "base": base, "margin": margin}


PREDICTED = {"shape": "Cordate", "apex": "Acute", "base": "Auriculate", "margin": "Entire"}
CONFIDENT = {"shape": 0.9, "apex": 0.9, "base": 0.9, "margin": 0.9}


def test_exact_match_wins():
    rules = [
        _rule("G1"),
        _rule("G2", shape="Ovate", apex="Obtuse", base="Cuneate", margin="Crenate"),
    ]
    results = match_group(PREDICTED, CONFIDENT, rules=rules)

    assert results[0]["code"] == "G1"
    assert results[0]["matched"] == 4


def test_more_matches_beats_a_higher_confidence_sum():
    """Why ranking is count-first rather than score-first.

    At the default conf_th of 0.6 the two orderings can never disagree: a
    matched trait always contributes at least 0.6, so more matches also means
    a higher sum. They only come apart once the threshold is lowered, which
    the test bench lets an operator do.
    """
    rules = [
        _rule("A", margin="Crenate"),                 # matches shape, apex, base
        _rule("B", shape="Ovate", apex="Obtuse"),     # matches base, margin
    ]
    probs = {"shape": 0.35, "apex": 0.35, "base": 0.95, "margin": 0.95}

    results = match_group(PREDICTED, probs, rules=rules, conf_th=0.3)
    by_code = {r["code"]: r for r in results}

    assert by_code["A"]["matched"] == 3
    assert by_code["B"]["matched"] == 2
    # B has the bigger confidence sum, yet A wins on matched count.
    assert by_code["B"]["score"] > by_code["A"]["score"]
    assert results[0]["code"] == "A"


def test_confidence_breaks_a_tie_between_equal_matches():
    rules = [_rule("G1"), _rule("G2", name="อีกกลุ่ม")]
    # Identical rules: same matched count, so the score decides — and with
    # identical scores the code decides.
    results = match_group(PREDICTED, CONFIDENT, rules=rules)
    assert [r["code"] for r in results] == ["G1", "G2"]


def test_matched_trait_adds_its_confidence_to_the_score():
    rules = [_rule("G1", apex="Obtuse", base="Cuneate", margin="Crenate")]
    results = match_group(PREDICTED, {"shape": 0.8, "apex": 0.9,
                                      "base": 0.9, "margin": 0.9}, rules=rules)
    assert results[0]["matched"] == 1
    assert results[0]["score"] == 0.8


def test_low_confidence_trait_counts_half_even_when_it_matches():
    rules = [_rule("G1", apex="Obtuse", base="Cuneate", margin="Crenate")]
    results = match_group(
        PREDICTED,
        {"shape": 0.4, "apex": 0.9, "base": 0.9, "margin": 0.9},
        rules=rules, conf_th=0.6,
    )
    assert results[0]["matched"] == 0.5
    assert results[0]["score"] == 0.5


def test_low_confidence_trait_counts_half_even_when_it_does_not_match():
    """An unsure model must not rule a group out."""
    rules = [_rule("G1", shape="Lanceolate", apex="Obtuse",
                   base="Cuneate", margin="Crenate")]
    results = match_group(
        PREDICTED,
        {"shape": 0.4, "apex": 0.9, "base": 0.9, "margin": 0.9},
        rules=rules, conf_th=0.6,
    )
    assert results[0]["matched"] == 0.5


def test_confident_mismatch_scores_nothing():
    rules = [_rule("G1", shape="Lanceolate", apex="Obtuse",
                   base="Cuneate", margin="Crenate")]
    results = match_group(PREDICTED, CONFIDENT, rules=rules)
    assert results[0]["matched"] == 0
    assert results[0]["score"] == 0


def test_matching_is_case_insensitive():
    rules = [_rule("G1")]
    lowered = {k: v.lower() for k, v in PREDICTED.items()}
    results = match_group(lowered, CONFIDENT, rules=rules)
    assert results[0]["matched"] == 4


def test_missing_trait_does_not_raise():
    rules = [_rule("G1")]
    results = match_group({"shape": "Cordate"}, {"shape": 0.9}, rules=rules)
    # The other three traits are absent from probs, so they are not flagged
    # uncertain and simply do not match.
    assert results[0]["matched"] == 1


def test_top_n_caps_the_result_list():
    rules = [_rule(f"G{i}") for i in range(1, 6)]
    results = match_group(PREDICTED, CONFIDENT, rules=rules, top_n=3)
    assert len(results) == 3


def test_empty_rule_base_returns_empty_list():
    assert match_group(PREDICTED, CONFIDENT, rules=[]) == []
