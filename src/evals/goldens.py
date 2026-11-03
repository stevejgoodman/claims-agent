"""Turn claims into DeepEval goldens, and agent Runs into test cases."""

import json
from dataclasses import asdict

from deepeval.dataset import Golden
from deepeval.test_case import LLMTestCase, ToolCall

from agent.agent_factory import AgentConfig
from agent.data import Dataset
from agent.runner import Run

PAYOUT_TOOL = "calculate_payout"


def make_golden(claim: dict, dataset: Dataset) -> Golden:
    """The handler's decision as the expected output. An approval should have used the payout tool."""
    label = dataset.decisions[claim["claim_id"]]
    return Golden(
        name=claim["claim_id"],
        input=json.dumps(claim),
        expected_output=json.dumps({k: label[k] for k in ("decision", "payout", "reason", "needs")}),
        additional_metadata={
            "claim_id": claim["claim_id"],
            "tiny": label["tiny"],
            "expected_tools": [PAYOUT_TOOL] if label["decision"] == "approve" else [],
        },
    )


def make_goldens(claims: list[dict], dataset: Dataset) -> list[Golden]:
    return [make_golden(claim, dataset) for claim in claims]


def to_test_case(run: Run, golden: Golden, config: AgentConfig) -> LLMTestCase:
    """One test case carrying everything the metrics need: the answer, the handler's decision and the tool calls."""
    answer = {
        **{k: getattr(run, k) for k in ("decision", "payout", "rationale", "cost", "retrieved")},
        "used_memory": config.use_memory,
    }
    return LLMTestCase(
        name=golden.name,
        input=golden.input,
        actual_output=json.dumps(answer),
        expected_output=golden.expected_output,
        tools_called=[ToolCall(name=call["name"], input_parameters=call["args"]) for call in run.tools_called],
        expected_tools=[ToolCall(name=name) for name in golden.additional_metadata["expected_tools"]],
    )
