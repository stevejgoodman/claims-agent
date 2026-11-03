"""Loading the claims, decisions, policies and customers, and splitting them into a dev set and a held-out set."""

import json
import random
from dataclasses import dataclass
from pathlib import Path

DATA = Path(__file__).resolve().parents[2] / "data"
EVALS = DATA / "evals"

# BASE and HELD_OUT policies and customers both live in the top-level files; ids are unique across all files
SOURCES = {
    "BASE": ("claims.json", "decisions.json", "policies.json", "customers.json"),
    "HARDER": ("harder/claims.json", "harder/decisions.json", "harder/policies.json", "harder/customers.json"),
    "EVEN_HARDER": ("even_harder/claims.json", "even_harder/decisions.json", "even_harder/policies.json", "even_harder/customers.json"),
    "HELD_OUT": ("held_out_claims.json", "held_out_decisions.json", "policies.json", "customers.json"),
}


@dataclass
class Dataset:
    claims: list[dict]
    decisions: dict[str, dict]  # claim_id -> the handler's decision
    policies: dict[str, dict]  # policy_id -> policy record
    customers: dict[str, dict]  # customer_id -> customer record


def _read(name: str) -> list[dict]:
    return json.loads((EVALS / name).read_text(encoding="utf-8"))


def load_all() -> Dataset:
    """Every claim from every source, each tagged with the source it came from."""
    claims, decisions, policies, customers = [], {}, {}, {}
    for source, (claims_file, decisions_file, policies_file, customers_file) in SOURCES.items():
        claims += [{**claim, "source": source} for claim in _read(claims_file)]
        for d in _read(decisions_file):
            decisions[d["claim_id"]] = {**d, "source": source}
        for p in _read(policies_file):
            policies[p["policy_id"]] = p
        for c in _read(customers_file):
            customers[c["customer_id"]] = c

    # A policy or customer takes the source of the claim that uses it (BASE and HELD_OUT share files).
    # A customer with claims in two sources keeps the source of the last one.
    for claim in claims:
        policies[claim["policy_id"]] = {**policies[claim["policy_id"]], "source": claim["source"]}
        customers[claim["customer_id"]] = {**customers[claim["customer_id"]], "source": claim["source"]}
    return Dataset(claims, decisions, policies, customers)


def split_ab(dataset: Dataset, seed: int = 42) -> tuple[list[dict], list[dict]]:
    """Two equal-sized sets of claims with a similar source mix. Set A is for development, set B is held back."""
    rng = random.Random(seed)
    set_a, set_b = [], []
    for source in SOURCES:
        group = [c for c in dataset.claims if c["source"] == source]
        rng.shuffle(group)
        set_a += group[::2]
        set_b += group[1::2]

    # Dealing per source can leave the sets uneven, so rebalance to equal size
    rng.shuffle(set_a)
    rng.shuffle(set_b)
    while len(set_a) > len(set_b):
        set_b.append(set_a.pop())
    while len(set_b) > len(set_a):
        set_a.append(set_b.pop())
    return set_a, set_b


def load_knowledge() -> list[dict]:
    """The facts about items and events that the agent can look up in its memory."""
    return json.loads((DATA / "knowledge.json").read_text(encoding="utf-8"))
