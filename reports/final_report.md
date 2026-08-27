# Day 10 Reliability Final Report

## Metrics Summary

| Metric | Value |
|---|---:|
| total_requests | 300 |
| availability | 0.9933 |
| error_rate | 0.0067 |
| latency_p50_ms | 272.14 |
| latency_p95_ms | 313.91 |
| latency_p99_ms | 318.8 |
| fallback_success_rate | 0.9726 |
| cache_hit_rate | 0.6167 |
| circuit_open_count | 8 |
| recovery_time_ms | 2408.5190296173096 |
| estimated_cost | 0.049526 |
| estimated_cost_saved | 0.185 |

## Chaos Scenarios

| Scenario | Status |
|---|---|
| primary_timeout_100 | pass: requests=100==100; availability=0.9800>=0.90; fallback_successes=35>0; circuit_opens=5>0 |
| primary_flaky_50 | pass: requests=100==100; availability=1.0000>=0.95; circuit_opens=3>0 |
| all_healthy | pass: requests=100==100; availability=1.0000>=0.98 |
| cache_comparison | pass: cost 0.050590->0.019510 (-61.4%); hit_rate 0.0000->0.6300; circuit_opens 2->1 |

## Analysis TODO(student)

Explain what failed, why the fallback path worked or did not work, and what you would change before production.