"""Run an agent config over a set of claims and score it. Used by the notebook and by pytest."""

import json

import pandas as pd

from agent.agent_factory import AgentConfig, build_agent
from agent.data import Dataset
from agent.runner import run_claim

from .goldens import make_goldens, to_test_case
from .metrics import RightMetric
from .scoring import score_table


def run_goldens(agent, goldens, dataset: Dataset, trace: bool = False) -> list:
    """Run the agent once on every golden and return a test case for each."""
    callbacks = None
    if trace:
        from deepeval.integrations.langchain import CallbackHandler

        callbacks = [CallbackHandler()]
    test_cases = []
    for golden in goldens:
        run = run_claim(agent, json.loads(golden.input), dataset, callbacks)
        print(f"{golden.name}: {run.decision} ${run.payout:,.2f}")
        test_cases.append(to_test_case(run, golden, agent.config))
    return test_cases


def run_experiment(name: str, config: AgentConfig, claims: list[dict], dataset: Dataset, n_times: int = 1) -> pd.DataFrame:
    """Run a config on the claims n_times, score each run with DeepEval, and return the claims of every run in one table."""
    from deepeval import evaluate
    from deepeval.evaluate.configs import AsyncConfig
    from deepeval.tracing.context import current_span_context, current_trace_context

    goldens = make_goldens(claims, dataset)
    agent = build_agent(config, dataset)
    tables = []
    for run in range(1, n_times + 1):
        # A trace left open by an earlier cell or a run that was stopped would break this run
        current_span_context.set(None)
        current_trace_context.set(None)
        test_cases = run_goldens(agent, goldens, dataset, trace=True)
        evaluate(
            test_cases=test_cases,
            metrics=[RightMetric()],
            identifier=f"{name}, run {run}",
            async_config=AsyncConfig(run_async=False),
        )
        tables.append(score_table(test_cases).assign(run=run))
    return pd.concat(tables, ignore_index=True)
