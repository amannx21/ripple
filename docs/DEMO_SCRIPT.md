# 2-minute demo script

Setup before recording: `make reproduce` has run (results/ exists), then `make app` and open http://localhost:8501.
Keep the sidebar on **Offline (scored dataset)** unless you are showing live mode.

**The three headline numbers** come from results/scorecard.json: held-out seeds 1000–1499, n = 500, SYNTHETIC data, α = 0.15 tuned on seeds 0–99.

| Metric | FreshRoute | FIFO baseline |
|---|---|---|
| Spoilage (% of stock) | **5.31 %** | 34.37 % |
| Demand fulfilled | **61.87 %** | 52.01 % |
| Net revenue per dispatch cycle | **₹26.29 L** | ₹10.31 L |

Constraint violations per plan: 0.00 for FreshRoute against 16.94 for FIFO, counted by the independent checker.
Counting diverted stock as loss too, the figures are 28.11 % against 34.37 %. Say this out loud, because it is the honest comparison.

---

**0:00 to 0:15 · Home.** "Shelf life isn't a number, it's a distribution that shrinks in the heat. FreshRoute samples it on
every route and plans ship, discount, hold or divert under a risk limit." Point at the **SYNTHETIC DATA** badge.

**0:15 to 0:35 · 1 Scorecard.** Read the three numbers above. Point at the **HELD-OUT · 500 scenarios** badge and at
"α chosen on tuning seeds 0–99 only". Scroll to the paired-difference table: the 95 % CI, and the planner is better in 100 % of scenarios on
net revenue. Mention that the deterministic optimizer ablation makes 18.6 % spoilage, so modeling uncertainty is what
does the work.

**0:35 to 0:55 · 2 Plan.** Show the green **0 violations ✓ (independent checker)** badge. Point at the dispatch queue: rank 1 leaves first. Read one
reason string aloud. Scroll to the SaR gauges: red batches have no destination inside α, so they are diverted.

**0:55 to 1:10 · 6 Explain.** Open one DISCOUNT card (a mango batch). Read the counterfactual: "not sending it to X →
… expected net −₹…". Expand **Discount timing**: markdown now beats waiting a day.

**1:10 to 1:30 · 4 Simulator & Replan.** Click **🔥 Heatwave** (+6 °C). Show "What changed", the old plan's violations
under the new conditions, and the realized comparison table, where the replanned plan beats the stale plan.

**1:30 to 1:45 · 3 Live Map & Weather.** Switch the sidebar to **Live geography**. Show the routes coloured by heat and the
**LIVE / CACHED / FALLBACK** badge with its timestamp. Then show the cooler-hours departure bar chart.

**1:45 to 2:00 · 8 Data & Assumptions.** Show the cost-model tab (REPLACE WITH ORGANIZER COST MODEL) and the
Assumptions tab. Close with: "Drop the organizer files into data/organizer, run make reproduce, and every number
updates. Nothing is hard-coded."
