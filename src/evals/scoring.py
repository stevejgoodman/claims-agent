"""Scores for the agent's decisions against the handlers'. Pure functions over answers and tables, no LLM."""

import json

import pandas as pd

DECISIONS = ["approve", "decline", "refer"]
WRONGFUL_DECLINE = 1500
NEEDLESS_REFERRAL = 50


def is_right(answer, expected):
    """True when the decision matches the handler's and the payout is within a dollar."""
    return answer["decision"] == expected["decision"] and abs(answer["payout"] - expected["payout"]) <= 1


def mistake_cost(answer, expected):
    """What the agent's answer costs UnderCover, against the handler's decision."""
    if answer["decision"] == expected["decision"]:
        return abs(answer["payout"] - expected["payout"]) if answer["decision"] == "approve" else 0
    if answer["decision"] == "decline":
        return WRONGFUL_DECLINE
    if answer["decision"] == "approve":
        return answer["payout"]
    return NEEDLESS_REFERRAL


def found_fact(answer, expected):
    """For a claim that needs a fact from the knowledge base, whether the agent's searches found it.

    None when the claim needs no fact, or the agent has no memory to search."""
    if not expected["needs"] or not answer.get("used_memory", True):
        return None
    return all(fact in answer["retrieved"] for fact in expected["needs"])


def kappa(table):
    """Cohen's kappa for the agent's decisions against the handlers'."""
    observed = (table["agent"] == table["handler"]).mean()
    agent_share = table["agent"].value_counts(normalize=True)
    handler_share = table["handler"].value_counts(normalize=True)
    expected = sum(agent_share.get(decision, 0) * handler_share.get(decision, 0) for decision in DECISIONS)
    return (observed - expected) / (1 - expected)


def show_confusion_matrix(table, title):
    """Draw the confusion matrix, with the handler's decision on each row and the agent's in each column."""
    import plotly.express as px

    matrix = pd.crosstab(table["handler"], table["agent"]).reindex(index=DECISIONS, columns=DECISIONS, fill_value=0)
    fig = px.imshow(matrix, text_auto=True, color_continuous_scale="Blues", labels={"x": "Agent", "y": "Handler", "color": "Claims"}, title=title)
    fig.show()


def score_table(test_cases):
    """One row for each claim in a run: the agent's answer beside the handler's, with the scores."""
    rows = []
    for test_case in test_cases:
        answer = json.loads(test_case.actual_output)
        expected = json.loads(test_case.expected_output)
        claim = json.loads(test_case.input)
        rows.append({
            "claim_id": test_case.name,
            "source": claim.get("source"),
            "claim_type": claim.get("claim_type"),
            "amount_claimed": claim.get("amount_claimed"),
            "handler": expected["decision"],
            "agent": answer["decision"],
            "handler_payout": expected["payout"],
            "agent_payout": answer["payout"],
            "right": is_right(answer, expected),
            "mistake_cost": mistake_cost(answer, expected),
            "found_fact": found_fact(answer, expected),
            "llm_cost": answer["cost"],
            "tools_called": [t.name for t in (test_case.tools_called or [])],
            "rationale": answer["rationale"],
        })
    return pd.DataFrame(rows).sort_values("claim_id")


def run_scores(table):
    """The scores for a whole run."""
    agent_approvals = table[table["agent"] == "approve"]
    handler_approvals = table[table["handler"] == "approve"]
    return {
        "decision_accuracy": (table["agent"] == table["handler"]).mean(),
        "accuracy": table["right"].mean(),
        "precision": (agent_approvals["handler"] == "approve").mean(),
        "recall": (handler_approvals["agent"] == "approve").mean(),
        "kappa": kappa(table),
        "cost_of_mistakes": table["mistake_cost"].sum(),
        "straight_through": (table["agent"] != "refer").mean(),
        "found_fact": table["found_fact"].dropna().astype(float).mean(),
        "llm_cost": table["llm_cost"].sum(),
    }


def average_scores(table):
    """The scores for each run of an experiment, averaged over the runs."""
    return pd.DataFrame([run_scores(run_table) for _, run_table in table.groupby("run")]).mean()
