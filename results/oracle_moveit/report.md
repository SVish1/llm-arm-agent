## Benchmark results: oracle (scripted) on `moveit`

- Episodes: 15 (15 tasks x 1 trials)
- **Success rate: 100.0%**
- Agent self-report matches ground truth: 100.0%
- Mean LLM calls / episode: 0.0; mean tool calls: 3.4; tool error rate: 0.0%
- Mean tokens / episode: in 0 (+0 cached), out 0

| Latency (s) | mean | p50 | p90 | max |
|---|---:|---:|---:|---:|
| Episode wall-clock | 15.59 | 13.48 | 38.85 | 46.69 |
| LLM time / episode | 0.00 | 0.00 | 0.00 | 0.00 |
| Tool execution / episode | 15.55 | 13.45 | 38.81 | 46.66 |
| Time to first action | 0.00 | 0.00 | 0.00 | 0.00 |
| Motion execution (sim/robot time) | 15.02 | 13.00 | 37.49 | 45.13 |
| Episode at real-time robot speed | 15.59 | 13.48 | 38.85 | 46.69 |

| Category | Success |
|---|---:|
| motion | 100% |
| multi-step | 100% |
| pick-place | 100% |
| robustness | 100% |
| spatial | 100% |

| Task | Category | Instruction | Success | Mean time (s) | LLM calls |
|---|---|---|---:|---:|---:|
| T01_home | motion | Move the arm back to its home position. | 1/1 | 1.9 | 0.0 |
| T02_raise | motion | Raise the gripper by 10 centimetres. | 1/1 | 1.7 | 0.0 |
| T03_hover | motion | Hover the gripper about 10 cm above the red cube. | 1/1 | 2.8 | 0.0 |
| T04_pick | pick-place | Pick up the red cube. | 1/1 | 7.1 | 0.0 |
| T05_to_zone | pick-place | Put the blue cube on the right pad. | 1/1 | 15.8 | 0.0 |
| T06_stack | pick-place | Stack the red cube on top of the green cube. | 1/1 | 16.7 | 0.0 |
| T07_relative | spatial | Place the yellow cube 10 cm to the left of the blue cube. | 1/1 | 13.5 | 0.0 |
| T08_closest | spatial | Put the cube that is closest to the robot onto the left pad. | 1/1 | 16.3 | 0.0 |
| T09_sort | multi-step | Put the red cube on the left pad and the blue cube on the right pad. | 1/1 | 27.2 | 0.0 |
| T10_tower | multi-step | Build a tower on the left pad: blue cube at the bottom, then the green cube, then the red cube on top. | 1/1 | 46.7 | 0.0 |
| T11_swap | multi-step | Swap the positions of the red cube and the green cube. | 1/1 | 44.9 | 0.0 |
| T12_clear_zone | multi-step | Clear the left pad by moving everything on it onto the right pad. Stacking is fine. | 1/1 | 29.8 | 0.0 |
| T13_lift_hold | pick-place | Lift the green cube to about 20 cm above the table and keep holding it. | 1/1 | 9.3 | 0.0 |
| T14_missing_object | robustness | Pick up the purple cube. | 1/1 | 0.1 | 0.0 |
| T15_unreachable | robustness | Put the red cube at x = 1.5 m, y = 0. | 1/1 | 0.1 | 0.0 |
