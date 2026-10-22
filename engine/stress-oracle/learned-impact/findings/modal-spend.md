# Modal spend ledger (learned impact)

Cap $2.00 for the 2026-10-22 Modal trial (`CAP_USD` in `modal/train_modal.py`; ~$8 more is
earmarked for the viability training, which raises the cap deliberately).

Rows are appended by `train_modal.py`: *estimate* is the worst case checked against the cap
before the call (hard timeout + 120 s start-up allowance, ×1.10); *cost* is charged on the
local wall time of the call (≥ the container's billed seconds) × (GPU + 4 cores + 8 GiB) at
modal.com/pricing = `modal billing rates` (2026-10-22): L4 $0.000222/s, L40S $0.000542/s,
core $0.0000131/s, GiB $0.00000222/s. CPU-only helpers (`probe`, `upload`) use 1 core + 2 GiB.
The reconciliation against `modal billing report` is at the end.

| date (UTC) | job | gpu | estimate | wall s | $/s | cost | note |
|---|---|---|---|---|---|---|---|
| 2026-10-22 05:21 | probe, first try (image build ~60 s; result failed to unpickle torch's version type locally) | cpu | $0.0000 | 71 | 0.000018 | $0.0013 | added by hand: raised before `record` |
| 2026-10-22 05:23 | probe (image build + env_info) | cpu | $0.0000 | 6 | 0.000018 | $0.0001 | {"python": "3.12.1", "torch": "2.14.1+cu130", "torch_cuda": "13.0", "cudnn": 92400, "numpy": "2.5.4"} |
| 2026-10-22 05:24 | upload sub100 | cpu | $0.0000 | 19 | 0.000018 | $0.0003 | remote hash missing |
| 2026-10-22 05:24 | upload sub100 | cpu | $0.0000 | 4 | 0.000018 | $0.0001 | remote hash ok |
| 2026-10-22 05:24 | upload sub100 | cpu | $0.0000 | 6 | 0.000018 | $0.0001 | remote hash ok |
| 2026-10-22 05:25 | trial l4-w64-pf | L4 | $0.1350 | 41 | 0.000292 | $0.0121 | rc 1; parity OK; 0.00 steps/s steady; 0 steps |
| 2026-10-22 05:29 | trial l4-w64-pf | L4 | $0.1350 | 183 | 0.000292 | $0.0534 | rc 0; parity OK; 2.58 steps/s steady; 300 steps |
| 2026-10-22 05:32 | trial l40s-w64-pf | L40S | $0.2828 | 116 | 0.000612 | $0.0711 | rc 0; parity OK; 4.13 steps/s steady; 300 steps |
| 2026-10-22 05:34 | trial l4-w64-nopf | L4 | $0.1350 | 125 | 0.000292 | $0.0366 | rc 0; parity OK; 6.04 steps/s steady; 300 steps |
| 2026-10-22 05:36 | trial l40s-w64-nopf | L40S | $0.2828 | 84 | 0.000612 | $0.0516 | rc 0; parity OK; 8.91 steps/s steady; 300 steps |
| 2026-10-22 05:41 | trial l4-w128-pf | L4 | $0.1350 | 201 | 0.000292 | $0.0587 | rc 0; parity OK; 2.35 steps/s steady; 300 steps; 4 cores, 8 GiB |
| 2026-10-22 05:45 | trial l4-w64-pf-x3 | L4 | $0.1923 | 219 | 0.000336 | $0.0735 | rc 0; parity OK; 2.28 + 2.21 + 2.34 steps/s steady; 300 steps; 6 cores, 16 GiB |
| 2026-10-22 05:48 | trial l40s-w64-pf-x3 | L40S | $0.3753 | 126 | 0.000656 | $0.0828 | rc 0; parity OK; 3.90 + 3.70 + 3.98 steps/s steady; 300 steps; 6 cores, 16 GiB |
| 2026-10-22 05:53 | trial l4-w128-pf-x3 | L4 | $0.1923 | 268 | 0.000336 | $0.0901 | rc 0; parity OK; 1.72 + 1.65 + 1.74 steps/s steady; 300 steps; 6 cores, 16 GiB |
| 2026-10-22 05:56 | trial l40s-w128-pf-x3 | L40S | $0.3753 | 171 | 0.000656 | $0.1121 | rc 0; parity OK; 2.75 + 2.64 + 2.76 steps/s steady; 300 steps; 6 cores, 16 GiB |
| 2026-10-22 06:01 | trial l4-w64-pf-x4 | L4 | $0.2174 | 235 | 0.000380 | $0.0893 | rc 0; parity OK; 2.07 + 1.98 + 2.08 + 2.23 steps/s steady; 300 steps; 8 cores, 24 GiB |

**Ledger total: $0.733** of the $2.00 cap (16 rows; the trial stopped there, with $1.27 unspent).

## Reconciliation with Modal's billing

`modal billing report --for today --resolution h --show-resources --json`, at 2026-10-22
06:05 UTC, summed over the app `learned-impact-train`: **$0.7175 billed**. By resource: L4
$0.2754, L40S $0.2622, CPU $0.1273, memory $0.0525, network egress $0.0001. The ledger's
wall-time charge ($0.733) is 2% above the bill, which confirms it as a safe upper bound.
The image build is billed too, but it is small: about $0.002 for the one ~60 s torch-layer build.
