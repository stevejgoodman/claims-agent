"""Offline tests for the extra checks in evals/checks.py. No model calls."""

import pandas as pd
import pytest

from agent.payout import APPROVAL_LIMIT, work_out_payout, Line
from agent.prompts import make_user_prompt
from evals import checks
from evals.checks import PAYOUT_TOOL, RECORD_TOOL


@pytest.fixture(scope="module")
def claim(dataset):
    return dataset.claims[0]  # CLM-1001, a trip delay with receipts


def call(name, **args):
    return {"name": name, "args": args}


def payout_call(claim, amount=245.0):
    return call(PAYOUT_TOOL, policy_id=claim["policy_id"], benefit="trip_delay",
                lines=[{"description": "Hotel", "amount": amount, "kind": "expense"}])


def worked_out(dataset, claim, amount=245.0):
    return work_out_payout(dataset.policies[claim["policy_id"]], "trip_delay", [Line(description="Hotel", amount=amount, kind="expense")])["payout"]


# --- 1. Business rules -----------------------------------------------------------------------------------------------


def test_clean_answer_has_no_rule_violations(claim):
    assert checks.rule_violations({"decision": "approve", "payout": 100.0}, claim) == []
    assert checks.rule_violations({"decision": "refer", "payout": 0.0}, claim) == []


@pytest.mark.parametrize(
    ("answer", "text"),
    [
        ({"decision": "decline", "payout": 50.0}, "should pay nothing"),
        ({"decision": "refer", "payout": 10.0}, "should pay nothing"),
        ({"decision": "approve", "payout": 0.0}, "pays nothing"),
        ({"decision": "approve", "payout": 9999.0}, "more than"),
    ],
)
def test_rule_violations_are_named(claim, answer, text):
    assert any(text in p for p in checks.rule_violations(answer, claim))


def test_payout_over_sign_off_limit_where_the_handler_paid_less(claim):
    big = {**claim, "amount_claimed": 5000.0}
    answer = {"decision": "approve", "payout": APPROVAL_LIMIT + 1}
    assert checks.rule_violations(answer, big) == []  # without the handler's decision there's nothing to compare
    assert any("sign-off" in p for p in checks.rule_violations(answer, big, {"payout": 400.0}))
    assert checks.rule_violations(answer, big, {"payout": 1500.0}) == []  # the handler paid it too


def test_handlers_approve_over_the_limit_in_the_data(dataset):
    """Why the over-limit rule is a comparison with the handler and not a blanket 'refer everything over $1000'."""
    assert any(d["decision"] == "approve" and d["payout"] > APPROVAL_LIMIT for d in dataset.decisions.values())


# --- 2. Tool sequence ------------------------------------------------------------------------------------------------


def test_good_sequence(dataset, claim):
    tools = [call("get_policy", policy_id=claim["policy_id"]), payout_call(claim),
             call(RECORD_TOOL, claim_id=claim["claim_id"], decision="approve", payout=worked_out(dataset, claim), rationale="x")]
    assert checks.tool_sequence_problems(tools, claim, dataset.policies) == []


def test_record_decision_must_happen_once_and_last(dataset, claim):
    record = call(RECORD_TOOL, claim_id="x", decision="refer", payout=0.0, rationale="x")
    assert any("0 times" in p for p in checks.tool_sequence_problems([call("get_policy")], claim, dataset.policies))
    assert any("2 times" in p for p in checks.tool_sequence_problems([record, record], claim, dataset.policies))
    assert any("not the last" in p for p in checks.tool_sequence_problems([record, call("get_policy")], claim, dataset.policies))


def test_recorded_payout_must_match_the_payout_tool(dataset, claim):
    tools = [payout_call(claim), call(RECORD_TOOL, claim_id="x", decision="approve", payout=worked_out(dataset, claim) + 50, rationale="x")]
    assert any("works out" in p for p in checks.tool_sequence_problems(tools, claim, dataset.policies))


def test_payout_tool_must_use_the_claims_policy(dataset, claim):
    wrong = payout_call(claim)
    wrong["args"]["policy_id"] = "POL-2002"
    tools = [wrong, call(RECORD_TOOL, claim_id="x", decision="approve", payout=1.0, rationale="x")]
    assert any("not the claim's" in p for p in checks.tool_sequence_problems(tools, claim, dataset.policies))


# --- 3. Unwanted calls -----------------------------------------------------------------------------------------------


def test_payout_tool_on_a_claim_that_is_not_paid():
    assert checks.unwanted_tool_calls([call(PAYOUT_TOOL)], expected_tools=[]) != []
    assert checks.unwanted_tool_calls([call("get_policy")], expected_tools=[]) == []
    assert checks.unwanted_tool_calls([call(PAYOUT_TOOL)], expected_tools=[PAYOUT_TOOL]) == []


# --- 4. Stability ----------------------------------------------------------------------------------------------------


def runs_table(*rows):
    return pd.DataFrame(rows, columns=["claim_id", "run", "agent", "agent_payout"])


def test_flip_rate():
    stable = runs_table(("A", 1, "approve", 100.0), ("A", 2, "approve", 100.5), ("B", 1, "refer", 0.0), ("B", 2, "refer", 0.0))
    assert checks.flip_rate(stable) == 0
    flipped = runs_table(("A", 1, "approve", 100.0), ("A", 2, "refer", 0.0), ("B", 1, "refer", 0.0), ("B", 2, "refer", 0.0))
    assert checks.flip_rate(flipped) == 0.5
    payout_moves = runs_table(("A", 1, "approve", 100.0), ("A", 2, "approve", 140.0))
    assert checks.flip_rate(payout_moves) == 1


# --- 5. Retrieval ----------------------------------------------------------------------------------------------------

KNOWLEDGE = [{"topic": "events", "fact": "Tropical Storm Marisol forms."}, {"topic": "events", "fact": "Lisbon strike on July 14."}]


def test_retrieval_precision():
    both = "Tropical Storm Marisol forms.\nLisbon strike on July 14."
    assert checks.retrieval_precision(both, ["Marisol"], KNOWLEDGE) == 0.5
    assert checks.retrieval_precision("Tropical Storm Marisol forms.", ["Marisol"], KNOWLEDGE) == 1
    assert checks.retrieval_precision("nothing useful", ["Marisol"], KNOWLEDGE) is None
    assert checks.retrieval_precision("Tropical Storm Marisol forms.", ["history"], KNOWLEDGE) is None  # not a memory fact


def test_searched_when_needed():
    assert checks.searched_when_needed([call("search_memory")], ["Marisol"]) is True
    assert checks.searched_when_needed([call("get_policy")], ["Marisol"]) is False
    assert checks.searched_when_needed([call("get_policy")], []) is None


# --- 6. Citations ----------------------------------------------------------------------------------------------------


def test_cited_sections():
    assert checks.cited_sections("Per section 3 and section 10.") == {3, 10}
    assert checks.cited_sections("See sections 3 and 10, and §7.") == {3, 7, 10}
    assert checks.cited_sections("No citation here, but 3 nights.") == set()


def test_wording_sections_match_the_real_wording():
    wording = open("data/policy_wording.md", encoding="utf-8").read()
    assert checks.wording_sections(wording) == set(range(1, 13))


def test_ungrounded_citations():
    wording = open("data/policy_wording.md", encoding="utf-8").read()
    assert checks.ungrounded_citations("Covered under section 3.", wording) == []
    assert checks.ungrounded_citations("Covered under section 99.", wording) == ["Cites section 99, which is not in the policy wording"]


# --- 7. Efficiency ---------------------------------------------------------------------------------------------------


def test_efficiency_counts_repeated_calls():
    tools = [call("get_policy", policy_id="P1"), call("get_policy", policy_id="P1"), call("get_policy", policy_id="P2")]
    assert checks.efficiency(tools, 0.02) == {"tool_calls": 3, "repeated_calls": 1, "cost": 0.02}


def test_efficiency_problems():
    assert checks.efficiency_problems([call("a", n=i) for i in range(3)], 0.01) == []
    assert len(checks.efficiency_problems([call("a", n=i) for i in range(20)], 0.5)) == 2
    assert any("repeated" in p for p in checks.efficiency_problems([call("a"), call("a")], 0.0))


# --- 10. Slices ------------------------------------------------------------------------------------------------------


def test_amount_band():
    assert [checks.amount_band(a) for a in (10, 250, 999.99, 1000, 5000)] == ["under $250", "$250 to $1000", "$250 to $1000", "over $1000", "over $1000"]


def test_slice_scores_find_the_weak_slice():
    table = pd.DataFrame({"source": ["BASE"] * 4 + ["HARDER"] * 4, "right": [True] * 4 + [True, False, False, False],
                          "mistake_cost": [0] * 4 + [0, 50, 50, 50]})
    scores = checks.slice_scores(table, "source")
    assert scores.loc["BASE", "accuracy"] == 1
    assert scores.loc["HARDER", "cost_of_mistakes"] == 150
    assert checks.weakest_slice(table, "source") == ("HARDER", 0.25)
    assert checks.weakest_slice(table, "source", min_claims=10) is None


# --- 11. Adversarial and edge-case claims ----------------------------------------------------------------------------


def test_claims_around_the_sign_off_limit(claim):
    amounts = [c["amount_claimed"] for c in checks.around_limit(claim)]
    assert amounts == [APPROVAL_LIMIT - 0.01, APPROVAL_LIMIT, APPROVAL_LIMIT + 0.01]
    assert claim["amount_claimed"] != APPROVAL_LIMIT  # the original is not changed


def test_missing_field_and_missing_documents(claim, dataset):
    assert "incident_date" not in checks.without_field(claim, "incident_date")
    assert checks.without_documents(claim)["documents"] == []
    assert claim["documents"]  # the original is not changed
    make_user_prompt(checks.without_field(claim, "story"), dataset.policies[claim["policy_id"]])  # the prompt still builds


@pytest.mark.parametrize("where", ["story", "document"])
def test_injection_reaches_the_prompt_and_is_detected(claim, dataset, where):
    attacked = checks.with_injection(claim, where)
    assert checks.INJECTION in make_user_prompt(attacked, dataset.policies[claim["policy_id"]])
    assert checks.INJECTION not in str(claim)
    assert checks.followed_injection({"decision": "approve", "payout": 99999.0})
    assert not checks.followed_injection({"decision": "refer", "payout": 0.0})


def test_duplicate_claim_has_a_new_id(claim):
    assert checks.duplicate_of(claim, "CLM-DUP")["claim_id"] == "CLM-DUP"
    assert checks.duplicate_of(claim, "CLM-DUP")["story"] == claim["story"]


# --- 8, 9. Judge -----------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(("reply", "verdict"), [("PASS. Supported.", True), ("fail: invented a receipt", False), ("Maybe", None), ("", None)])
def test_judge_verdict(reply, verdict):
    assert checks.judge_verdict(reply) is verdict


def test_judge_prompts_carry_the_answer(claim):
    seen = []

    def judge(prompt):
        seen.append(prompt)
        return "PASS fine"

    answer = {"decision": "approve", "payout": 245.0, "rationale": "Covered under section 3.", "retrieved": ""}
    assert checks.judge_faithfulness(judge, claim, answer) is True
    assert checks.judge_consistency(judge, answer) is True
    assert all("Covered under section 3." in p for p in seen)


def test_judge_agreement_exposes_an_always_pass_judge():
    human = [True, True, True, True, False, False]
    always_pass = checks.judge_agreement([True] * 6, human)
    assert always_pass["accuracy"] == pytest.approx(4 / 6)
    assert always_pass["true_negative_rate"] == 0  # it never catches a bad rationale
    good = checks.judge_agreement([True, True, True, True, False, False], human)
    assert good["accuracy"] == 1 and good["true_negative_rate"] == 1
    assert checks.judge_agreement([None, True], [True, True])["no_verdict"] == 1
