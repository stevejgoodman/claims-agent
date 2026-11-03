import json

import pandas as pd
import pytest
from deepeval.dataset import Golden

from agent.agent_factory import AgentConfig
from agent.runner import Run
from evals.goldens import PAYOUT_TOOL, make_golden, to_test_case
from evals.metrics import RightMetric, payout_tool_metric
from evals.scoring import found_fact, is_right, kappa, mistake_cost, run_scores, score_table


def answer(decision, payout=0.0, **kw):
    return {"decision": decision, "payout": payout, **kw}


def test_is_right_allows_a_dollar():
    assert is_right(answer("approve", 100.5), answer("approve", 100))
    assert not is_right(answer("approve", 102), answer("approve", 100))
    assert not is_right(answer("decline"), answer("refer"))


@pytest.mark.parametrize(
    ("agent", "handler", "cost"),
    [
        (answer("approve", 90), answer("approve", 100), 10),
        (answer("decline"), answer("approve", 100), 1500),
        (answer("approve", 100), answer("decline"), 100),
        (answer("refer"), answer("approve", 100), 50),
        (answer("decline"), answer("decline"), 0),
    ],
)
def test_mistake_cost(agent, handler, cost):
    assert mistake_cost(agent, handler) == cost


def test_found_fact():
    needs = {"needs": ["Marisol"]}
    assert found_fact({"retrieved": "Tropical Storm Marisol"}, needs)
    assert found_fact({"retrieved": ""}, needs) is False
    assert found_fact({"retrieved": ""}, {"needs": []}) is None
    assert found_fact({"retrieved": "", "used_memory": False}, needs) is None  # no memory: not applicable


def test_kappa_is_one_for_perfect_agreement():
    table = pd.DataFrame({"agent": ["approve", "decline", "refer"], "handler": ["approve", "decline", "refer"]})
    assert kappa(table) == 1


def make_run(claim_id, decision, payout=0.0, tools=(), retrieved=""):
    return Run(claim_id, decision, payout, "because", 0.01, [{"name": t, "args": {}} for t in tools], retrieved)


def test_golden_expects_payout_tool_only_for_approvals(dataset):
    by_decision = {}
    for claim in dataset.claims:
        golden = make_golden(claim, dataset)
        by_decision.setdefault(dataset.decisions[claim["claim_id"]]["decision"], set()).add(
            tuple(golden.additional_metadata["expected_tools"])
        )
    assert by_decision["approve"] == {(PAYOUT_TOOL,)}
    assert by_decision["decline"] == {()}


def golden_for(decision, payout, needs=()):
    expected = {"decision": decision, "payout": payout, "reason": "", "needs": list(needs)}
    return Golden(
        name="C1", input="{}", expected_output=json.dumps(expected),
        additional_metadata={"expected_tools": [PAYOUT_TOOL] if decision == "approve" else []},
    )


def test_test_case_scores_end_to_end():
    case = to_test_case(make_run("C1", "approve", 100, tools=["get_policy", PAYOUT_TOOL]), golden_for("approve", 100), AgentConfig())
    metric = RightMetric()
    assert metric.measure(case) == 1

    table = score_table([case]).assign(run=1)
    scores = run_scores(table)
    assert scores["accuracy"] == 1
    assert scores["cost_of_mistakes"] == 0


def test_tool_metric_catches_a_missing_payout_call():
    golden = golden_for("approve", 100)
    metric = payout_tool_metric()
    assert metric.measure(to_test_case(make_run("C1", "approve", 100, tools=[PAYOUT_TOOL]), golden, AgentConfig())) == 1
    assert metric.measure(to_test_case(make_run("C1", "refer"), golden, AgentConfig())) == 0
