"""Live evals: they call the real model and cost money. Run with `pytest -m eval`.

EVAL_CONFIG   name of the config in evals.configs (default baseline)
EVAL_MAX      number of claims to run (default 5; 0 for all of them)
EVAL_SET      A (development, default) or B (held out; only for a final check)
"""

import os

import pytest
from deepeval import assert_test

from agent.agent_factory import build_agent
from evals.configs import get_config
from evals.experiment import run_goldens
from evals.goldens import make_goldens
from evals.metrics import RightMetric, payout_tool_metric
from evals.scoring import run_scores, score_table

pytestmark = pytest.mark.eval

# Run-level thresholds. Tighten them as the agent improves.
MIN_ACCURACY = 0.6
MAX_MISTAKE_COST_PER_CLAIM = 500


@pytest.fixture(scope="module")
def config():
    return get_config(os.getenv("EVAL_CONFIG", "baseline"))


@pytest.fixture(scope="module")
def test_cases(dataset, sets, config):
    claims = sets[1] if os.getenv("EVAL_SET", "A").upper() == "B" else sets[0]
    limit = int(os.getenv("EVAL_MAX", "5"))
    claims = claims[:limit] if limit else claims
    agent = build_agent(config, dataset)
    return run_goldens(agent, make_goldens(claims, dataset), dataset)


def test_each_claim(test_cases):
    failures = []
    for case in test_cases:
        metrics = [RightMetric()]
        if case.expected_tools:  # only approvals need the payout tool
            metrics.append(payout_tool_metric())
        try:
            assert_test(case, metrics)
        except AssertionError as error:
            failures.append(f"{case.name}: {error}")
    assert not failures, "\n".join(failures)


def test_run_scores(test_cases):
    scores = run_scores(score_table(test_cases).assign(run=1))
    assert scores["accuracy"] >= MIN_ACCURACY
    assert scores["cost_of_mistakes"] / len(test_cases) <= MAX_MISTAKE_COST_PER_CLAIM


def test_memory_is_searched_for_claims_that_need_a_fact(test_cases, config):
    if not config.use_memory:
        pytest.skip("No memory in this config")
    table = score_table(test_cases)
    found = table["found_fact"].dropna()
    if found.empty:
        pytest.skip("No claim in this sample needs a fact")
    assert found.all(), f"Facts missed on {list(table.loc[found.index[~found.astype(bool)], 'claim_id'])}"


# --- Extra checks (see evals/checks.py). Same sample of claims, no extra model calls except the last three tests. ---

import json  # noqa: E402
from pathlib import Path  # noqa: E402

from agent.data import load_knowledge  # noqa: E402
from agent.runner import run_claim  # noqa: E402
from evals import checks  # noqa: E402
from evals.metrics import business_rules_metric, case_parts, citations_metric, judge_passes, tool_sequence_metric  # noqa: E402

WORDING = (Path(__file__).resolve().parents[1] / "data" / "policy_wording.md").read_text(encoding="utf-8")
JUDGE_LABELS = Path(__file__).resolve().parents[1] / "data" / "evals" / "judge_labels.json"


parts = case_parts


def assert_metric_on_every_case(test_cases, metric_factory):
    """Fresh metric for each case, so one case's score never leaks into the next."""
    failures = []
    for case in test_cases:
        try:
            assert_test(case, [metric_factory()])
        except AssertionError as error:
            failures.append(f"{case.name}: {error}")
    assert not failures, "\n".join(failures)


def check_every_claim(test_cases, check):
    problems = [f"{case.name}: {p}" for case in test_cases for p in check(case, *parts(case))]
    assert not problems, "\n".join(problems)


def test_business_rules(test_cases):
    assert_metric_on_every_case(test_cases, business_rules_metric)


def test_tool_sequence(test_cases, dataset):
    assert_metric_on_every_case(test_cases, lambda: tool_sequence_metric(dataset.policies))

def test_no_payout_tool_on_claims_that_are_not_paid(test_cases):
    check_every_claim(
        test_cases,
        lambda case, claim, answer, expected, tools: checks.unwanted_tool_calls(tools, [t.name for t in case.expected_tools]),
    )


def test_retrieval_precision_and_search(test_cases, config):
    if not config.use_memory:
        pytest.skip("No memory in this config")
    knowledge, problems = load_knowledge(), []
    for case in test_cases:
        _, answer, expected, tools = parts(case)
        if checks.searched_when_needed(tools, expected["needs"]) is False:
            problems.append(f"{case.name}: needed {expected['needs']} but never searched its memory")
        precision = checks.retrieval_precision(answer["retrieved"], expected["needs"], knowledge)
        if expected["needs"] and precision is not None and precision < 0.5:
            problems.append(f"{case.name}: only {precision:.0%} of the retrieved facts were relevant")
    assert not problems, "\n".join(problems)


def test_citations_exist_in_the_wording(test_cases):
    assert_metric_on_every_case(test_cases, lambda: citations_metric(WORDING))


def test_efficiency(test_cases):
    check_every_claim(test_cases, lambda case, claim, answer, expected, tools: checks.efficiency_problems(tools, answer["cost"]))


def test_no_slice_is_much_worse_than_the_rest(test_cases):
    table = score_table(test_cases).assign(amount_band=lambda t: t["amount_claimed"].map(checks.amount_band))
    for by in ("source", "claim_type", "amount_band"):
        weakest = checks.weakest_slice(table, by)
        if weakest:
            assert weakest[1] >= MIN_SLICE_ACCURACY, f"Weakest {by}: {weakest[0]} at {weakest[1]:.0%}"


MIN_SLICE_ACCURACY = 0.4


def test_injected_instruction_is_not_followed(dataset, sets, config):
    claim = sets[0][0]
    agent = build_agent(config, dataset)
    for where in ("story", "document"):
        run = run_claim(agent, checks.with_injection(claim, where), dataset)
        assert not checks.followed_injection({"decision": run.decision, "payout": run.payout}), f"Followed the instruction in the {where}"


def test_claims_near_the_sign_off_limit_never_pay_over_it_unreviewed(dataset, sets, config):
    claim = next(c for c in sets[0] if dataset.decisions[c["claim_id"]]["decision"] == "approve")
    agent = build_agent(config, dataset)
    for variant in checks.around_limit(claim):
        run = run_claim(agent, variant, dataset)
        assert not checks.rule_violations({"decision": run.decision, "payout": run.payout}, variant), variant["amount_claimed"]


MIN_JUDGE_AGREEMENT = 0.85


def test_judge_is_trustworthy_before_its_scores_are_used(dataset):
    """Needs about 30 hand-labelled rationales. Each label is {claim_id, decision, payout, rationale, retrieved, rationale_ok},
    where rationale_ok is true when the explanation is both faithful and consistent."""
    if not JUDGE_LABELS.exists():
        pytest.skip(f"Label about 30 rationales by hand in {JUDGE_LABELS.name} first")
    labels = json.loads(JUDGE_LABELS.read_text(encoding="utf-8"))
    claims = {claim["claim_id"]: claim for claim in dataset.claims}
    judged = [judge_passes(claims[label["claim_id"]], label) for label in labels]
    agreement = checks.judge_agreement(judged, [label["rationale_ok"] for label in labels])
    assert agreement["accuracy"] >= MIN_JUDGE_AGREEMENT, agreement
    assert agreement["true_negative_rate"] >= MIN_JUDGE_AGREEMENT, f"The judge lets bad rationales through: {agreement}"


MAX_FLIP_RATE = 0.2


def test_decisions_are_stable_across_runs(dataset, sets, config):
    """Runs the claims EVAL_RUNS times (default 2), so it costs that many times a normal run. Uses at most 5 claims."""
    from evals.experiment import run_experiment

    table = run_experiment("stability", config, sets[0][:5], dataset, n_times=int(os.getenv("EVAL_RUNS", "2")))
    assert checks.flip_rate(table) <= MAX_FLIP_RATE
