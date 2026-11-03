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
    assert found.all(), f"Facts missed on {list(table.loc[found.index[~found], 'claim_id'])}"
