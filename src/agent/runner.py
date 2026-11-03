"""Run an agent on one claim. This is the only place the agent is invoked."""

from dataclasses import dataclass, field

from .agent_factory import Agent
from .data import Dataset
from .memory import SEARCH_TOOL
from .prompts import make_user_prompt


@dataclass
class Run:
    claim_id: str
    decision: str
    payout: float
    rationale: str
    cost: float  # what the model calls cost, in US dollars
    tools_called: list[dict] = field(default_factory=list)  # [{"name", "args"}] in call order
    retrieved: str = ""  # everything the memory searches returned
    messages: list = field(default_factory=list)


def run_claim(agent: Agent, claim: dict, dataset: Dataset, callbacks: list | None = None) -> Run:
    claim_id = claim["claim_id"]
    agent.ledger.pop(claim_id, None)  # decisions are final, so a repeat run needs a clean slate
    result = agent.graph.invoke(
        {"messages": [{"role": "user", "content": make_user_prompt(claim, dataset.policies[claim["policy_id"]])}]},
        config={"callbacks": callbacks or []},
    )
    messages = result["messages"]
    answer = agent.ledger.get(claim_id, {"decision": "refer", "payout": 0.0, "rationale": "Nothing recorded"})
    return Run(
        claim_id=claim_id,
        **answer,
        cost=sum((m.response_metadata.get("token_usage") or {}).get("cost", 0.0) for m in messages if m.type == "ai"),
        tools_called=[{"name": call["name"], "args": call["args"]} for m in messages if m.type == "ai" for call in m.tool_calls],
        retrieved="\n".join(str(m.text) for m in messages if m.type == "tool" and m.name == SEARCH_TOOL),
        messages=messages,
    )
