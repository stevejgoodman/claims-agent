"""DeepEval metrics. Each reads a test case built by goldens.to_test_case."""

import json
from typing import Callable

from deepeval.metrics import BaseMetric, FaithfulnessMetric, GEval, ToolCorrectnessMetric
from deepeval.test_case import LLMTestCase, SingleTurnParams

from . import checks
from .scoring import found_fact, is_right, mistake_cost


class RightMetric(BaseMetric):
    """Test case metric: 1 when the agent's decision matches the handler's and the payout is within a dollar."""

    def __init__(self):
        self.threshold = 1
        self.async_mode = False

    def measure(self, test_case, *args, **kwargs):
        answer = json.loads(test_case.actual_output)
        expected = json.loads(test_case.expected_output)
        self.score = int(is_right(answer, expected))
        self.success = self.score >= self.threshold
        self.reason = f"Agent: {answer['decision']} ${answer['payout']:,.2f}. Handler: {expected['decision']} ${expected['payout']:,.2f}. Cost of mistake: ${mistake_cost(answer, expected):,.2f}"
        found = found_fact(answer, expected)
        if found is not None:
            self.reason += f". Found the fact: {found}"
        return self.score

    async def a_measure(self, test_case, *args, **kwargs):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return "Right"


def payout_tool_metric():
    """1 when every expected tool was called. Extra tools (get_policy, record_decision, ...) are fine."""
    return ToolCorrectnessMetric(threshold=1.0, async_mode=False)


# --- Deterministic checks as metrics -----------------------------------------------------------------------------------


def case_parts(case) -> tuple[dict, dict, dict, list[dict]]:
    """A test case as plain dicts: the claim, the agent's answer, the handler's decision and the tool calls."""
    tools = [{"name": t.name, "args": t.input_parameters or {}} for t in case.tools_called]
    return json.loads(case.input), json.loads(case.actual_output), json.loads(case.expected_output), tools


class ProblemsMetric(BaseMetric):
    """Wraps a check that returns a list of problems: 1 when it finds none, and the problems as the reason."""

    def __init__(self, name: str, find: Callable[[LLMTestCase], list[str]]):
        self.threshold = 1
        self.async_mode = False
        self.name, self.find = name, find

    def measure(self, test_case, *args, **kwargs):
        problems = self.find(test_case)
        self.score = int(not problems)
        self.success = self.score >= self.threshold
        self.reason = "; ".join(problems) or "No problems"
        return self.score

    async def a_measure(self, test_case, *args, **kwargs):
        return self.measure(test_case)

    def is_successful(self):
        return self.success

    @property
    def __name__(self):
        return self.name


def business_rules_metric() -> ProblemsMetric:
    def find(case):
        claim, answer, expected, _ = case_parts(case)
        return checks.rule_violations(answer, claim, expected)

    return ProblemsMetric("Business rules", find)


def tool_sequence_metric(policies: dict[str, dict]) -> ProblemsMetric:
    def find(case):
        claim, _, _, tools = case_parts(case)
        return checks.tool_sequence_problems(tools, claim, policies)

    return ProblemsMetric("Tool sequence", find)


def citations_metric(wording: str) -> ProblemsMetric:
    def find(case):
        _, answer, _, _ = case_parts(case)
        return checks.ungrounded_citations(answer["rationale"], wording)

    return ProblemsMetric("Citations exist", find)


# --- The judges --------------------------------------------------------------------------------------------------------

JUDGE_MODEL = "anthropic/claude-sonnet-5.5"
FAITHFULNESS_THRESHOLD = 1.0  # every statement in the explanation must be supported
CONSISTENCY_THRESHOLD = 0.5

CONSISTENCY_CRITERIA = (
    "Does the explanation argue for the decision and payout stated above it? Fail it if it argues for a different "
    "decision, or gives reasons that do not lead to this one."
)


def judge_model():
    from deepeval.models import OpenRouterModel

    return OpenRouterModel(model=JUDGE_MODEL, temperature=0)


def faithfulness_case(claim: dict, answer: dict) -> LLMTestCase:
    """The explanation, to be checked against the claim and documents and whatever the agent retrieved from memory."""
    return LLMTestCase(
        input="Explain the decision on this claim.",
        actual_output=answer["rationale"],
        retrieval_context=[json.dumps(claim), answer.get("retrieved") or "none"],
    )


def consistency_case(answer: dict) -> LLMTestCase:
    """The decision and the explanation together, so the judge can see whether one supports the other."""
    text = f"Decision: {answer['decision']}, payout ${answer['payout']:,.2f}\nExplanation: {answer['rationale']}"
    return LLMTestCase(input="Explain the decision on this claim.", actual_output=text)


def faithfulness_metric(model=None) -> FaithfulnessMetric:
    return FaithfulnessMetric(threshold=FAITHFULNESS_THRESHOLD, model=model or judge_model(), include_reason=True, async_mode=False)


def consistency_metric(model=None) -> GEval:
    return GEval(
        name="Consistency",
        criteria=CONSISTENCY_CRITERIA,
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        threshold=CONSISTENCY_THRESHOLD,
        model=model or judge_model(),
        async_mode=False,
    )


def judge_passes(claim: dict, answer: dict, model=None) -> bool:
    """True when the explanation is both faithful and consistent. One label to compare with a person's."""
    model = model or judge_model()
    faithful, consistent = faithfulness_metric(model), consistency_metric(model)
    faithful.measure(faithfulness_case(claim, answer))
    consistent.measure(consistency_case(answer))
    return faithful.is_successful() and consistent.is_successful()
