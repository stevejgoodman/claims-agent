"""DeepEval metrics. Each reads a test case built by goldens.to_test_case."""

import json

from deepeval.metrics import BaseMetric, ToolCorrectnessMetric

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
