"""Build a claims agent from an AgentConfig. Each experiment is a different config."""

import os
from dataclasses import dataclass, field

from langchain.agents import create_agent
from langchain.agents.middleware import ToolRetryMiddleware
from langchain_openai import ChatOpenAI

from .data import Dataset, load_knowledge
from .memory import build_store, memory_tools
from .prompts import INSTRUCTIONS, build_system_prompt
from .tools import Ledger, make_tools

OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"


@dataclass(frozen=True)
class AgentConfig:
    model: str = "openai/gpt-6-luna"
    reasoning_effort: str = "low"
    use_memory: bool = True
    use_payout_tool: bool = True
    instructions: str = INSTRUCTIONS
    tool_retries: int = 3


@dataclass
class Agent:
    """A compiled agent graph with the config it was built from and the ledger its record_decision tool writes to."""

    graph: object
    config: AgentConfig
    ledger: Ledger = field(default_factory=Ledger)


def build_agent(config: AgentConfig, dataset: Dataset, llm=None) -> Agent:
    """llm replaces the OpenRouter model, for tests."""
    llm = llm or ChatOpenAI(
        model=config.model,
        base_url=OPENROUTER_BASE_URL,
        api_key=os.getenv("OPENROUTER_API_KEY"),
        reasoning_effort=config.reasoning_effort,
        timeout=120,
        max_retries=5,
    )
    ledger = Ledger()
    tools = make_tools(dataset.policies, ledger, config.use_payout_tool)
    kwargs = {}
    if config.use_memory:
        tools += memory_tools()
        kwargs["store"] = build_store(load_knowledge())
    graph = create_agent(
        model=llm,
        tools=tools,
        system_prompt=build_system_prompt(config.use_memory, config.use_payout_tool, config.instructions),
        middleware=[ToolRetryMiddleware(max_retries=config.tool_retries)],
        **kwargs,
    )
    return Agent(graph, config, ledger)
