from langchain_core.messages import AIMessage

from agent.agent_factory import AgentConfig, build_agent
from evals.configs import CONFIGS, get_config
from evals.experiment import run_goldens
from evals.goldens import make_goldens
from evals.scoring import score_table


def test_configs_registry():
    assert get_config("no_memory").use_memory is False
    assert all(c.use_memory for n, c in CONFIGS.items() if n != "no_memory")


def test_run_goldens_gives_a_scoreable_test_case_per_claim(dataset, sets, scripted_model):
    claim = sets[0][0]
    agent = build_agent(AgentConfig(use_memory=False), dataset, llm=scripted_model(AIMessage(content="nothing")))
    test_cases = run_goldens(agent, make_goldens([claim], dataset), dataset)
    table = score_table(test_cases)
    assert list(table["claim_id"]) == [claim["claim_id"]]
    assert table.loc[0, "agent"] == "refer"  # nothing recorded
    assert table["found_fact"].dropna().empty  # no memory: retrieval doesn't apply
