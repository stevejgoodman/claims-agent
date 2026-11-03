"""The agent's instructions and the prompt for one claim."""

import json

INSTRUCTIONS = """You are a claims handler at UnderCover Travel Insurance.
Decide the claim you are given, using the policy wording below and the customer's policy record.

Choose one decision:
- approve: the wording covers the claim and the required documents are attached. Work out the payout using section 10.
- decline: the wording does not cover the claim.
- refer: a colleague needs to look at it, because a required document is missing, or the story and the documents disagree, or the wording does not settle the question.

Use your tools to record your decision.
"""

# Only added when the agent has the memory tools
MEMORY_LINE = "When you are asked a question, use your tools to search your memory for facts about items or events.\n"

# Only added when the agent has the payout tool
PAYOUT_TOOL_LINE = "Use the payout tool to work out the payout on an approved claim, and record the payout it returns.\n"


def build_system_prompt(use_memory: bool = True, use_payout_tool: bool = True, instructions: str = INSTRUCTIONS) -> str:
    prompt = instructions
    if use_memory:
        prompt += MEMORY_LINE
    if use_payout_tool:
        prompt += PAYOUT_TOOL_LINE
    return prompt


def make_user_prompt(claim: dict, policy: dict) -> str:
    """The customer's policy record and the claim, as JSON."""
    return f"""# Policy record

{json.dumps(policy, indent=2)}

# Claim

{json.dumps(claim, indent=2)}
"""
