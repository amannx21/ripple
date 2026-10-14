# Baselines

The organizers have not yet supplied a baseline. If they do, it will be implemented as the official comparison, and Baseline A stays as an extra. All baselines are scored by the same simulator, cost model, scenarios and realized worlds as the planner, and checked by the same independent constraint checker with the same α.

## Baseline A: FIFO (official comparison until the organizers publish one)

Implemented in `src/freshroute/baseline.py::fifo`.

1. Sort batches **oldest first**: `age_days` descending, ties broken by `batch_id`.
2. For each batch, list destinations reachable by a lane from its hub that have a price for the commodity. Rank them by **price, highest first**; ties go to the shorter distance. A `nearest` variant ranks by distance instead.
3. **Point-estimate feasibility**, the spreadsheet check a planner would do:
   `estimated remaining life − (base travel hours + 3 handling hours)/24 ≥ minimum sellable life`
   and base travel time ≤ the delivery window. Shelf life is consumed at the reference temperature, so this check sees no heat and no uncertainty.
4. Ship `floor(min(batch kg left, remaining demand ÷ (1 − handling loss), lane truck capacity left, destination receiving capacity left))`, then move to the next destination. Stop when the batch is gone or no destination has room.
5. **No discounting and no diverting.** Whatever is not shipped is HELD at the hub.
6. Trucks per lane = ceil(kg on lane / truck capacity).

## Baseline B: greedy-by-price

Same rules as FIFO. Instead of iterating batches, it iterates (batch, destination) pairs in order of **price descending**, ties broken by oldest batch first.

## Ablation: deterministic optimizer

The same MILP as FreshRoute, solved on point estimates: all uncertainty set to zero, so every probability is 0 or 1. It isolates how much of the gain comes from modeling uncertainty rather than from optimizing.

## Why baselines show constraint violations

The checker evaluates every plan against the Monte Carlo risk table at the same α. A baseline shipment whose P(arrives sellable and on time) is below 1 − α counts as a **shelf_life** violation. A held batch that is unlikely to remain sellable counts as a **hold** violation. Stock, capacity and demand constraints are respected by construction in both baselines.
