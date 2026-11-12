"""Offline tests for the DeepEval metrics in evals/metrics.py. No model calls."""

import json

import pytest
from deepeval.test_case import LLMTestCase, ToolCall

from evals import metrics
from evals.checks import PAYOUT_TOOL, RECORD_TOOL


@pytest.fixture(scope="module")
def claim(dataset):
    return dataset.claims[0]


def case_for(claim, decision="approve", payout=245.0, rationale="Covered under section 3.", tools=()):
    return LLMTestCase(
        input=json.dumps(claim),
        actual_output=json.dumps({"decision": decision, "payout": payout, "rationale": rationale, "retrieved": ""}),
        expected_output=json.dumps({"decision": "approve", "payout": 245.0, "reason": "", "needs": []}),
        tools_called=[ToolCall(name=name, input_parameters=args) for name, args in tools],
    )


def test_problems_metric_passes_when_clean_and_explains_when_not(claim):
    assert metrics.business_rules_metric().measure(case_for(claim)) == 1
    bad = metrics.business_rules_metric()
    assert bad.measure(case_for(claim, decision="decline", payout=100.0)) == 0
    assert "should pay nothing" in bad.reason and not bad.is_successful()


def test_tool_sequence_metric_flags_a_missing_recording(claim, dataset):
    metric = metrics.tool_sequence_metric(dataset.policies)
    assert metric.measure(case_for(claim, tools=[(PAYOUT_TOOL, {"policy_id": claim["policy_id"]})])) == 0
    assert RECORD_TOOL in metric.reason


def test_citations_metric_flags_a_section_the_wording_lacks(claim):
    wording = "## 1. Intro\n## 2. Cover\n"
    assert metrics.citations_metric(wording).measure(case_for(claim, rationale="See section 2.")) == 1
    metric = metrics.citations_metric(wording)
    assert metric.measure(case_for(claim, rationale="See section 9.")) == 0
    assert "section 9" in metric.reason


def test_faithfulness_case_checks_the_explanation_against_the_claim_and_memory(claim):
    case = metrics.faithfulness_case(claim, {"rationale": "Hotel was paid.", "retrieved": "Fact: X"})
    assert case.actual_output == "Hotel was paid."
    assert claim["story"] in case.retrieval_context[0] and case.retrieval_context[1] == "Fact: X"
    assert metrics.faithfulness_case(claim, {"rationale": "r", "retrieved": ""}).retrieval_context[1] == "none"


def test_consistency_case_puts_the_decision_beside_the_explanation():
    case = metrics.consistency_case({"decision": "refer", "payout": 0.0, "rationale": "Needs a review."})
    assert "Decision: refer, payout $0.00" in case.actual_output and "Needs a review." in case.actual_output
