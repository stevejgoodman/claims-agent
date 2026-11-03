"""Named agent configs. The notebook and pytest (EVAL_CONFIG=<name>) both pick from here."""

from dataclasses import replace

from agent.agent_factory import AgentConfig

BASELINE = AgentConfig()

CONFIGS = {
    "baseline": BASELINE,
    "with_context": replace(BASELINE, use_context=True),
    "no_memory": replace(BASELINE, use_memory=False),
    "no_payout_tool": replace(BASELINE, use_payout_tool=False),
    "high_reasoning": replace(BASELINE, reasoning_effort="high"),
}


def get_config(name: str) -> AgentConfig:
    if name not in CONFIGS:
        raise KeyError(f"No config {name!r}. Choose from {sorted(CONFIGS)}")
    return CONFIGS[name]
