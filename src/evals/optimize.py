"""Tighten the agent's instructions with DeepEval's prompt optimizer (GEPA), scored by how often the agent matches the handler."""

import asyncio
import json
import random
from dataclasses import replace

from agent.agent_factory import AgentConfig, build_agent
from agent.data import Dataset
from agent.prompts import INSTRUCTIONS
from agent.runner import run_claim

from .goldens import make_goldens
from .metrics import RightMetric

OPTIMIZER_MODEL = "anthropic/claude-sonnet-5.5"  # rewrites the prompt; it never judges an answer, RightMetric does that


class OptimizerRightMetric(RightMetric):
    """RightMetric that also tells the optimizer why the handler decided as they did and what the agent said."""

    def measure(self, test_case, *args, **kwargs):
        score = super().measure(test_case)
        answer, expected = json.loads(test_case.actual_output), json.loads(test_case.expected_output)
        if not self.success:
            self.reason += f". Handler's reason: {expected['reason']} Agent's rationale: {answer['rationale']}"
        return score


def train_subset(claims: list[dict], size: int, seed: int = 0) -> tuple[list[dict], list[dict]]:
    """(train, unseen): a seeded sample of the claims to optimize on, and the claims left over."""
    train_ids = {c["claim_id"] for c in random.Random(seed).sample(claims, size)}
    return [c for c in claims if c["claim_id"] in train_ids], [c for c in claims if c["claim_id"] not in train_ids]


def make_model_callback(base: AgentConfig, dataset: Dataset):
    """The optimizer's model_callback: run the agent, with the prompt being tried as its instructions, on a golden's claim."""
    agents = {}  # one agent for each prompt, so the memory store is built once

    def run(prompt, golden) -> str:
        if prompt.text_template not in agents:
            agents[prompt.text_template] = build_agent(replace(base, instructions=prompt.text_template), dataset)
        claim = json.loads(golden.input)
        result = run_claim(agents[prompt.text_template], claim, dataset)
        return json.dumps({
            **{k: getattr(result, k) for k in ("decision", "payout", "rationale", "cost", "retrieved")},
            "used_memory": base.use_memory,
        })

    async def a_run(prompt, golden) -> str:
        return await asyncio.to_thread(run, prompt, golden)

    return a_run


def optimize_instructions(base: AgentConfig, claims: list[dict], dataset: Dataset, iterations: int = 6, minibatch_size: int = 8, pareto_size: int = 6, max_concurrent: int = 8):
    """Run GEPA from the default instructions on these claims. Returns the best instructions and the optimizer."""
    from deepeval.models import OpenRouterModel
    from deepeval.optimizer import PromptOptimizer
    from deepeval.optimizer.algorithms import GEPA
    from deepeval.optimizer.configs import AsyncConfig
    from deepeval.prompt import Prompt

    model = OpenRouterModel(model=OPTIMIZER_MODEL, temperature=0)
    optimizer = PromptOptimizer(
        model_callback=make_model_callback(base, dataset),
        metrics=[OptimizerRightMetric()],
        optimizer_model=model,
        algorithm=GEPA(iterations=iterations, minibatch_size=minibatch_size, pareto_size=pareto_size, random_seed=0, reflection_model=model, mutation_model=model),
        async_config=AsyncConfig(run_async=True, max_concurrent=max_concurrent),
    )
    # DeepEval 4.3.1 calls _accrue_tokens on the scorer and rewriter, which only have _accrue_cost
    for part in (optimizer.algorithm.scorer, optimizer.algorithm._rewriter):
        part._accrue_tokens = lambda input_tokens, output_tokens: None
    best = optimizer.optimize(prompt=Prompt(text_template=base.instructions), goldens=make_goldens(claims, dataset))
    return best.text_template, optimizer
