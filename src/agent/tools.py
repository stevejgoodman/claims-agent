"""The agent's tools. They close over the policies and a Ledger, so each agent has its own state."""

import json
from typing import Literal

from langchain_core.tools import tool

from .payout import Benefit, Line, Payout, work_out_payout


class Ledger(dict):
    """claim_id -> the decision the agent recorded: {"decision", "payout", "rationale"}."""


def make_tools(policies: dict[str, dict], ledger: Ledger, use_payout_tool: bool = True) -> list:
    """get_policy, record_decision and (optionally) calculate_payout, bound to these policies and this ledger."""

    # these would be database calls in a real-er system.
    @tool
    def get_policy(policy_id: str) -> str:
        """Look up a customer's policy record: the plan, the add-ons, the purchase date, the trip and the travelers."""
        policy = policies.get(policy_id)
        return json.dumps(policy, indent=2) if policy else f"There is no policy {policy_id}"

    @tool(parse_docstring=True)
    def record_decision(claim_id: str, rationale: str, decision: Literal["approve", "decline", "refer"], payout: float) -> str:
        """Record the decision on a claim in the claims system.

        Args:
            claim_id: The claim's id
            rationale: 2-3 sentences to explain the decision along with the relevant section of the policy
            decision: The decision on the claim
            payout: The amount to pay in US dollars, or 0 for a decline or a referral
        """
        if claim_id in ledger:
            return f"A decision is already recorded on {claim_id}, and decisions are final"
        if decision == "approve" and payout <= 0:
            return "An approval needs a payout above zero"
        ledger[claim_id] = {"decision": decision, "payout": payout, "rationale": rationale}
        return f"Recorded {decision} on {claim_id}"

    @tool(parse_docstring=True)
    def calculate_payout(policy_id: str, benefit: Benefit, lines: list[Line], paid_by_others: float = 0.0) -> Payout:
        """Work out the payout on an approved claim with UnderCover's payout rules.

        Args:
            policy_id: The customer's policy id
            benefit: The benefit the claim is paid under
            lines: The covered items and expenses, one for each line of the claim, with any excluded ones left out. A pair or a set counts as one item
            paid_by_others: Anything a carrier, hotel or other party has paid or refunded for the same loss

        Returns:
            The calculation, the payout and whether it needs sign-off
        """
        return work_out_payout(policies[policy_id], benefit, lines, paid_by_others)

    tools = [record_decision, get_policy]
    if use_payout_tool:
        tools.append(calculate_payout)
    return tools
