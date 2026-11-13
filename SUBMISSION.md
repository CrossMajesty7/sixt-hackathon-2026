# Submission: Project 4, Team "Kaini"

## Authors
**Team name:** Kaini

**Project:** Project 4: Right car, right station, next 14 days (Fleet Allocation Challenge)

## Problem and approach

**Problem.** Ten stations, four car categories, 14 days. Some stations have too many cars, others too few. We plan fleet additions, removals and transfers (driver or truck) to maximize profit = revenue − holding cost − addition cost − transfer cost. New demand is unknown and must be estimated from 8 weeks of completed rentals.

**Approach.** Three steps: build the data, forecast demand, optimize the plan with a mixed-integer model.

1. **Level 0 (data), `level0.py`.** For every station × category × day we compute committed car-days (open + booked rentals, clipped to the horizon), one-way arrivals and departures (a one-way car changes station the day after its return), and the station fleet. Start fleet = idle snapshot cars + on-rent cars at the pickup station of their rental. The result is compared with `data/check_values.csv`.
2. **Forecast, `forecast.py`.**
   - Demand to come = mean cars on rent per station, category and weekday (from completed rentals, unfolded to every rental day) minus the committed car-days of that day, floored at 0.
   - Base revenue per car-day = revenue ÷ rental days of each completed rental, averaged per station and category.
3. **Optimization (PuLP + CBC), one model per level that builds on the previous one:**
   - `level1.py`: integer `add` / `remove` per station, category, day. Fleet balance (R3/R4), fleet ≥ committed (R5), addition/removal caps per category, free capacity `max(0, max_utilization × fleet − committed)` (R6, modelled exactly with a binary variable).
   - `level2.py`: driver transfers as a time-space network with lead times, holding cost for cars in transit. Only station pairs where a driver transfer is not more expensive than a new car (28 of 90 pairs) are allowed, to keep the model solvable.
   - `level3.py`: upgrades to the next higher category (R7) and trucks with 8 seats (cost per truck, any category mix on the same route and day). The level 2 plan is passed as a warm start.

**Fare adjustment (price factors).** The revenue curve `q × price_factor(q)` is cut into buckets with a marginal revenue per car-day (Appendix C), so the solver can model it linearly. We check every curve: for `downtown / midsize` the marginal revenue is not decreasing (1.081, 0.906, 0.528, **1.224**, 0.856). A plain LP would fill the buckets in the wrong order and overstate revenue. For such curves we force the fill order with binary variables.

**Assumptions.**
- Demand per weekday equals the 8-week mean; the sum "committed + demand to come" stays equal to that mean, so far-away days have a larger demand to come.
- New rentals are round trips and do not change the station fleet.
- Base revenue is constant per station and category. In the data, the price per car-day does not depend on weekday or rental length (checked: weekday means differ by about 3 %, rental length by about 1 %).

## Models and why

| Model | Open-weight or closed | Where it runs | Main metric(s) | Cost and time per item | Chosen? |
| --- | --- | --- | --- | --- | --- |
| Claude Sonnet 5.5 | closed | API / Claude Code | Max input 1,000,000 / max output 128,000 tokens |  | used as coding assistant, not part of the plan |

## Results

**Local approximation (our own scorer, forecast used as the demand scenario, committed revenue left out because it is the same for every plan).** These are *not* scorer results.

| Plan | valid | profit | cars added / removed / transferred |
| --- | --- | --- | --- |
| do-nothing | yes | 3,008,619.77 | 0 / 0 / 0 |
| baseline | yes | 3,111,408.11 | 96 / 0 / 0 |
| level 1 | yes | 3,395,887.83 | 103 / 233 / 0 |
| level 2 | yes | 3,721,166.40 | 101 / 233 / 445 |
| level 3 / final | yes | 3,840,060.55 |  |

**Official scorer (feedback round).** TODO: copy the profit from each `*.result.md` and the leaderboard position here, next to the baseline profit. Plans are in `Project_4/plans/` (`level1_v1.csv`, `level2_v1.csv`, `level3_v1.csv`).

**What did not work / limits.**
- Our first level 3 run found no feasible solution within the time limit (CBC status −4). A corrected warm start was needed; the level 3 and level 2 solves can end at the time limit, so those plans are good but not proven near-optimal.
- Level 1 earns most of its gain by removing 233 cars. This depends fully on the forecast; if demand is underestimated, the removals cost revenue.
- The forecast has no spread (single mean per weekday) and does not correct the start-of-history effect (rentals that began before the 8 weeks are missing). The weekday mean is divided by 8, which assumes each weekday occurs exactly 8 times in the history.
- The local scorer uses the same rules as the model, so a misreading of a rule would not show up there. Only the official scorer is independent.
- Level 4 (robustness against forecast errors) is not done beyond the bucket-order check.
- Transfers are limited to 28 cheap pairs for drivers; long-distance transfers were never evaluated.

## How to run

Python 3.14 or newer. Packages: `pip install -r requirements.txt` (numpy, pandas, pandera, pulp with CBC). No environment variables or secrets.

The scripts expect the folders `fleetkit/` and `data/` next to them (as in `Project_4/`). Run from that folder:

```
python level0.py      # level0_output.csv, checks against data/check_values.csv
python forecast.py    # demand_forecast_output.csv (needs level0_output.csv)
python level1.py      # plans/level1_v1.csv
python level2.py      # plans/level2_v1.csv  (about 4 min)
python level3.py      # plans/level3_v1.csv  (up to about 7 min; uses plans/level2_v1.csv as warm start if present)
```

`level1.py` to `level3.py` compute the inputs themselves (they import `level0.py` and `forecast.py`), so only the order of the levels matters for the warm start and the comparison table. The plan files are written with `fleetkit.write_plan`; copy them to the team folder to get the official result.

## Demo

TODO: `submission/slides.pdf` (max 5 slides) or `submission/demo.mp4` (1 min, under 50 MB). The file must exist in the repo before the deadline.

## Next steps

- Replace the weekday mean by a forecast that uses `booking_date` (how early rentals are booked) and handles the start of the history, and add several demand scenarios so the plan is robust (level 4).
- Allow long-distance and truck transfers in a fuller search, and compare against the hidden-scenario mean, not one draw.
- For SIXT: the same model can run on a rolling 14-day horizon every day. Inputs are the fleet snapshot, open bookings and the rental history; the output is a list of additions, removals and transfers per station and category.
