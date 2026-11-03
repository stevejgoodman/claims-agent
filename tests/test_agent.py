import json

from langchain_core.messages import AIMessage

from agent.agent_factory import AgentConfig, build_agent
from agent.prompts import MEMORY_LINE, PAYOUT_TOOL_LINE, build_system_prompt
from agent.runner import run_claim
from agent.tools import Ledger, make_tools


def tool_by_name(tools, name):
    return next(t for t in tools if t.name == name)


def test_split_is_equal_disjoint_and_seeded(dataset, sets):
    set_a, set_b = sets
    assert len(set_a) == len(set_b)
    assert not {c["claim_id"] for c in set_a} & {c["claim_id"] for c in set_b}
    assert len(set_a) + len(set_b) == len(dataset.claims)
    assert [c["claim_id"] for c in set_a] == [c["claim_id"] for c in sets[0]]


def test_every_claim_has_a_policy_and_a_decision(dataset):
    for claim in dataset.claims:
        assert claim["policy_id"] in dataset.policies
        assert claim["claim_id"] in dataset.decisions


def test_record_decision_is_final_and_needs_a_payout_for_approvals(dataset):
    ledger = Ledger()
    record = tool_by_name(make_tools(dataset.policies, ledger), "record_decision")
    args = {"claim_id": "X", "rationale": "r", "decision": "approve", "payout": 0}
    assert "above zero" in record.invoke(args)
    assert "X" not in ledger
    assert record.invoke({**args, "payout": 10}) == "Recorded approve on X"
    assert "already recorded" in record.invoke({**args, "payout": 20})
    assert ledger["X"]["payout"] == 10


def test_ledgers_are_not_shared_between_agents(dataset):
    a, b = Ledger(), Ledger()
    tool_by_name(make_tools(dataset.policies, a), "record_decision").invoke(
        {"claim_id": "X", "rationale": "r", "decision": "refer", "payout": 0}
    )
    assert "X" in a
    assert "X" not in b


def test_get_policy(dataset):
    policy_id = next(iter(dataset.policies))
    get_policy = tool_by_name(make_tools(dataset.policies, Ledger()), "get_policy")
    assert json.loads(get_policy.invoke({"policy_id": policy_id}))["policy_id"] == policy_id
    assert "no policy" in get_policy.invoke({"policy_id": "nope"})


def test_payout_tool_is_optional(dataset):
    names = [t.name for t in make_tools(dataset.policies, Ledger(), use_payout_tool=False)]
    assert "calculate_payout" not in names


def test_prompt_lines_follow_the_config():
    assert MEMORY_LINE in build_system_prompt(use_memory=True, use_payout_tool=False)
    assert PAYOUT_TOOL_LINE not in build_system_prompt(use_memory=True, use_payout_tool=False)
    assert MEMORY_LINE not in build_system_prompt(use_memory=False)


def test_run_claim_reads_the_decision_and_tools_called(dataset, sets, scripted_model):
    claim = sets[0][0]
    record_call = {
        "name": "record_decision",
        "id": "1",
        "args": {"claim_id": claim["claim_id"], "rationale": "r", "decision": "refer", "payout": 0},
    }
    llm = scripted_model(AIMessage(content="", tool_calls=[record_call]), AIMessage(content="done"))
    agent = build_agent(AgentConfig(use_memory=False), dataset, llm=llm)

    run = run_claim(agent, claim, dataset)

    assert (run.decision, run.payout) == ("refer", 0)
    assert [c["name"] for c in run.tools_called] == ["record_decision"]
    assert run.retrieved == ""


def test_run_claim_with_nothing_recorded_refers(dataset, sets, scripted_model):
    agent = build_agent(AgentConfig(use_memory=False), dataset, llm=scripted_model(AIMessage(content="no")))
    run = run_claim(agent, sets[0][0], dataset)
    assert (run.decision, run.rationale) == ("refer", "Nothing recorded")
