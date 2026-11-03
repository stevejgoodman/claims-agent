import json
import random
from pathlib import Path


DATA = Path(__file__).parent.parent / "data"
EVALS = DATA / "evals"

# BASE and HELD_OUT policies and customers both live in the top-level files; ids are unique across all files
SOURCES = {
    "BASE": ("claims.json", "decisions.json", "policies.json", "customers.json"),
    "HARDER": ("harder/claims.json", "harder/decisions.json", "harder/policies.json", "harder/customers.json"),
    "EVEN_HARDER": ("even_harder/claims.json", "even_harder/decisions.json", "even_harder/policies.json", "even_harder/customers.json"),
    "HELD_OUT": ("held_out_claims.json", "held_out_decisions.json", "policies.json", "customers.json"),
}


def load(name, path=None):
    """Read one of the JSON files under data/evals."""
    return json.loads(((path or EVALS) / name).read_text(encoding="utf-8"))


def combine_datasets():
    """Merge every source into one claims list plus decisions, policies and customers keyed by id.

    Each record is tagged with the `source` it came from.
    """
    all_claims, all_decisions, all_policies, all_customers = [], {}, {}, {}
    for source, (claims_file, decisions_file, policies_file, customers_file) in SOURCES.items():
        for claim in load(claims_file):
            all_claims.append({**claim, "source": source})
        for d in load(decisions_file):
            all_decisions[d["claim_id"]] = {**d, "source": source}
        for p in load(policies_file):
            all_policies[p["policy_id"]] = p
        for c in load(customers_file):
            all_customers[c["customer_id"]] = c

    # A policy or customer takes the source of the claim that uses it (BASE and HELD_OUT share files).
    # A customer with claims in two sources keeps the source of the last one.
    for claim in all_claims:
        all_policies[claim["policy_id"]] = {**all_policies[claim["policy_id"]], "source": claim["source"]}
        all_customers[claim["customer_id"]] = {**all_customers[claim["customer_id"]], "source": claim["source"]}
    return all_claims, all_decisions, all_policies, all_customers


def split_claims(claims, seed=42):
    """Split claims 50-50, stratified by source so both halves have a similar source mix."""
    rng = random.Random(seed)
    set_a, set_b = [], []
    for source in SOURCES:
        group = [c for c in claims if c["source"] == source]
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


def split_files(claims_in_set, all_decisions, all_policies, all_customers):
    """The claims, decisions, policies and customers that belong to one set, as a dict of datasets."""
    return {
        "claims": claims_in_set,
        "decisions": {c["claim_id"]: all_decisions[c["claim_id"]] for c in claims_in_set},
        "policies": {c["policy_id"]: all_policies[c["policy_id"]] for c in claims_in_set},
        "customers": {c["customer_id"]: all_customers[c["customer_id"]] for c in claims_in_set},
    }


def create_splits(seed=42):
    """Return (eval_split, holdout_split), a 50-50 split of the combined datasets.

    Each split is a dict with keys "claims" (list), and "decisions", "policies",
    "customers" (dicts keyed by claim_id, policy_id and customer_id).
    """
    all_claims, all_decisions, all_policies, all_customers = combine_datasets()
    eval_claims, holdout_claims = split_claims(all_claims, seed)
    assert not {c["claim_id"] for c in eval_claims} & {c["claim_id"] for c in holdout_claims}
    return (
        split_files(eval_claims, all_decisions, all_policies, all_customers),
        split_files(holdout_claims, all_decisions, all_policies, all_customers),
    )
