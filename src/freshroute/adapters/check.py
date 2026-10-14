"""Validate the dataset that will be used: ``python -m freshroute.adapters.check``."""
from __future__ import annotations

import sys

from . import load_base_instance, organizer
from .tables import AdapterError


def main() -> int:
    which = "organizer" if organizer.organizer_data_present() else "synthetic"
    try:
        inst = load_base_instance(which)
    except AdapterError as exc:
        print(f"DATA ERROR ({which}):\n{exc}")
        return 1
    print(f"OK: {inst.source_label} data with {len(inst.batches)} batches, {len(inst.destinations)} destinations, "
          f"{len(inst.lanes)} lanes, {len(inst.prices)} prices, {len(inst.demand)} demand rows.")
    for n in inst.notes:
        print("note:", n)
    if which == "organizer":
        print(f"scenarios found: {len(organizer.load_scenarios())}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
