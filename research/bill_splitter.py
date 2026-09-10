"""
Cost-splitting logic (brief requirements E, F, G).

Each item is assigned to exactly one person. Additional charges (tax,
service charge) are distributed proportionally based on each person's
share of the item subtotal -- so someone who ordered more pays
proportionally more of the tax too, which is the fairest default split.
Rounding remainder is added to whichever person has the largest share,
so the per-person totals always sum EXACTLY to the bill total (brief
requirement G).
"""


def split_bill(items: list, assignments: dict, additional_charges: list, total: int) -> dict:
    """
    items: [{"name", "qty", "unit_price", "total_price"}, ...] (indices matter)
    assignments: {item_index: person_name}
    additional_charges: [{"label", "amount"}, ...]
    total: overall bill total (the ground truth to reconcile against)

    Returns: {person_name: {"items": [...], "item_subtotal": int,
                             "charge_share": int, "final_total": int}}
    """
    people = sorted(set(assignments.values()))
    if not people:
        return {}

    per_person = {p: {"items": [], "item_subtotal": 0} for p in people}

    for idx, item in enumerate(items):
        person = assignments.get(idx)
        if person is None:
            continue
        per_person[person]["items"].append(item)
        per_person[person]["item_subtotal"] += item["total_price"]

    overall_item_subtotal = sum(p["item_subtotal"] for p in per_person.values())
    total_charges = sum(c["amount"] for c in additional_charges)

    # Proportional charge share, largest-remainder method so shares sum
    # exactly to total_charges (avoids losing/gaining a few Rupiah to
    # independent rounding of each person's share).
    raw_shares = {}
    for p in people:
        proportion = (per_person[p]["item_subtotal"] / overall_item_subtotal) if overall_item_subtotal else 0
        raw_shares[p] = proportion * total_charges

    floor_shares = {p: int(raw_shares[p]) for p in people}
    remainder = total_charges - sum(floor_shares.values())
    # give leftover Rupiah, one each, to the people with the largest
    # fractional remainder first
    remainder_order = sorted(people, key=lambda p: raw_shares[p] - floor_shares[p], reverse=True)
    for p in remainder_order[:remainder]:
        floor_shares[p] += 1

    for p in people:
        per_person[p]["charge_share"] = floor_shares[p]
        per_person[p]["final_total"] = per_person[p]["item_subtotal"] + floor_shares[p]

    # Final reconciliation against the bill's stated total (handles the
    # case where subtotal+charges printed on the receipt don't perfectly
    # equal "total" due to receipt rounding) -- adjust the largest payer.
    computed_sum = sum(p["final_total"] for p in per_person.values())
    diff = total - computed_sum
    if diff != 0 and people:
        largest_payer = max(people, key=lambda p: per_person[p]["final_total"])
        per_person[largest_payer]["final_total"] += diff

    return per_person
