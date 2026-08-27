from __future__ import annotations

import json
import random
from pathlib import Path

from reliability_lab.cache import ResponseCache, SharedRedisCache
from reliability_lab.circuit_breaker import CircuitBreaker
from reliability_lab.config import LabConfig, ScenarioConfig
from reliability_lab.gateway import ReliabilityGateway
from reliability_lab.metrics import RunMetrics
from reliability_lab.providers import FakeLLMProvider


def load_queries(path: str | Path = "data/sample_queries.jsonl") -> list[str]:
    queries: list[str] = []
    for line in Path(path).read_text().splitlines():
        if not line.strip():
            continue
        queries.append(json.loads(line)["query"])
    return queries


def build_gateway(
    config: LabConfig,
    provider_overrides: dict[str, float] | None = None,
    rng: random.Random | None = None,
    cache_enabled: bool | None = None,
) -> ReliabilityGateway:
    providers = []
    for p in config.providers:
        fail_rate = provider_overrides.get(p.name, p.fail_rate) if provider_overrides else p.fail_rate
        providers.append(
            FakeLLMProvider(p.name, fail_rate, p.base_latency_ms, p.cost_per_1k_tokens, rng)
        )
    breakers = {
        p.name: CircuitBreaker(
            name=p.name,
            failure_threshold=config.circuit_breaker.failure_threshold,
            reset_timeout_seconds=config.circuit_breaker.reset_timeout_seconds,
            success_threshold=config.circuit_breaker.success_threshold,
        )
        for p in config.providers
    }
    cache: ResponseCache | SharedRedisCache | None = None
    use_cache = config.cache.enabled if cache_enabled is None else cache_enabled
    if use_cache:
        if config.cache.backend == "redis":
            cache = SharedRedisCache(
                config.cache.redis_url,
                config.cache.ttl_seconds,
                config.cache.similarity_threshold,
            )
        else:
            cache = ResponseCache(config.cache.ttl_seconds, config.cache.similarity_threshold)
    return ReliabilityGateway(providers, breakers, cache)


def calculate_recovery_time_ms(gateway: ReliabilityGateway) -> float | None:
    """Average time from a circuit opening to it closing again, in milliseconds."""
    recoveries: list[float] = []
    for breaker in gateway.breakers.values():
        opened_ts: float | None = None
        for entry in breaker.transition_log:
            if entry["to"] == "open":
                opened_ts = float(entry["ts"])
            elif entry["to"] == "closed" and opened_ts is not None:
                recoveries.append((float(entry["ts"]) - opened_ts) * 1000.0)
                opened_ts = None
    if not recoveries:
        return None
    return sum(recoveries) / len(recoveries)


def scenario_rng(config: LabConfig, scenario_name: str) -> random.Random | None:
    """Per-scenario RNG derived from the config seed, or None when unseeded.

    Deriving from "{seed}:{name}" keeps each scenario independent: adding or
    reordering scenarios does not shift the random stream of the others.
    """
    if config.seed is None:
        return None
    return random.Random(f"{config.seed}:{scenario_name}")


def run_scenario(
    config: LabConfig,
    queries: list[str],
    scenario: ScenarioConfig,
    rng: random.Random | None = None,
) -> RunMetrics:
    """Run a single named chaos scenario and collect its metrics."""
    if rng is None:
        rng = scenario_rng(config, scenario.name)
    picker = rng if rng is not None else random
    gateway = build_gateway(
        config, scenario.provider_overrides or None, rng, scenario.cache_enabled
    )
    metrics = RunMetrics()

    for _ in range(config.load_test.requests):
        prompt = picker.choice(queries)
        result = gateway.complete(prompt)

        metrics.total_requests += 1
        metrics.estimated_cost += result.estimated_cost

        if result.cache_hit:
            metrics.cache_hits += 1
            metrics.estimated_cost_saved += 0.001
            metrics.successful_requests += 1
        elif result.route == "fallback":
            metrics.fallback_successes += 1
            metrics.successful_requests += 1
        elif result.route == "static_fallback":
            metrics.static_fallbacks += 1
            metrics.failed_requests += 1
        else:
            metrics.successful_requests += 1

        if result.latency_ms > 0:
            metrics.latencies_ms.append(result.latency_ms)

    metrics.circuit_open_count = sum(
        1
        for breaker in gateway.breakers.values()
        for entry in breaker.transition_log
        if entry["to"] == "open"
    )
    metrics.recovery_time_ms = calculate_recovery_time_ms(gateway)
    return metrics


def evaluate_scenario(
    config: LabConfig, scenario: ScenarioConfig, result: RunMetrics
) -> tuple[bool, str]:
    """Judge a scenario against criteria derived from what it actually breaks.

    The criteria adapt to the scenario's own overrides rather than being keyed on
    its name, so a newly added scenario is graded without touching this function.
    """
    def availability_at_least(floor: float) -> tuple[bool, str]:
        return (
            result.availability >= floor,
            f"availability={result.availability:.4f}>={floor:.2f}",
        )

    def breaker_tripped() -> tuple[bool, str]:
        return result.circuit_open_count > 0, f"circuit_opens={result.circuit_open_count}>0"

    checks: list[tuple[bool, str]] = [
        (
            result.total_requests == config.load_test.requests,
            f"requests={result.total_requests}=={config.load_test.requests}",
        )
    ]

    overrides = scenario.provider_overrides
    dead = [name for name, rate in overrides.items() if rate >= 1.0]
    degraded = [name for name, rate in overrides.items() if 0.0 < rate < 1.0]

    if scenario.cache_enabled is False:
        # Control run: the point is to prove the cache really is off.
        checks.append((result.cache_hits == 0, f"cache_hits={result.cache_hits}==0"))
        checks.append(availability_at_least(0.95))
    elif dead:
        # A provider is fully dead: the breaker must notice, traffic must shift, and
        # availability must hold near the ceiling the surviving providers allow.
        #
        # That ceiling is derived from config rather than hardcoded: with every dead
        # provider removed, the best any request can do is the most reliable survivor,
        # i.e. 1 - min(fail_rate).  The 0.05 margin absorbs sampling noise — at 100
        # requests only ~40 reach a provider, so a 5% failure rate lands anywhere
        # between 0 and 5 static fallbacks on an ordinary run.
        survivors = [p.fail_rate for p in config.providers if p.name not in dead]
        ceiling = 1.0 - min(survivors) if survivors else 0.0
        checks.append(availability_at_least(max(0.0, ceiling - 0.05)))
        checks.append(
            (result.fallback_successes > 0, f"fallback_successes={result.fallback_successes}>0")
        )
        checks.append(breaker_tripped())
    elif degraded:
        # Partial failures: the breaker must notice, and the chain must hold the SLO.
        # Note the floor is availability, not static_fallbacks==0: with a backup that
        # itself fails 5% of the time, an occasional double failure is expected
        # behaviour, not a defect, and an absolute-zero check would be wishful.
        checks.append(availability_at_least(0.95))
        checks.append(breaker_tripped())
    else:
        # Healthy baseline: both providers up, so cache + two tiers should clear 0.98.
        checks.append(availability_at_least(0.98))

    failed = [label for ok, label in checks if not ok]
    if failed:
        return False, "; ".join(failed)
    return True, "; ".join(label for _, label in checks)


def compare_cache(config: LabConfig, queries: list[str]) -> str:
    """Run the healthy baseline twice — cache on, cache off — and report the delta.

    Both runs share one seed, so provider failures and token counts are identical and
    the only variable is the cache.  This is a control experiment, so its counters are
    deliberately NOT folded into the combined metrics.
    """
    baseline = ScenarioConfig(name="cache_comparison", description="cache on vs off")

    with_cache = run_scenario(
        config, queries, baseline.model_copy(update={"cache_enabled": True}),
        scenario_rng(config, "cache_comparison"),
    )
    without_cache = run_scenario(
        config, queries, baseline.model_copy(update={"cache_enabled": False}),
        scenario_rng(config, "cache_comparison"),
    )

    cost_delta = (
        (with_cache.estimated_cost - without_cache.estimated_cost) / without_cache.estimated_cost
        if without_cache.estimated_cost
        else 0.0
    )
    verdict = "pass" if with_cache.estimated_cost < without_cache.estimated_cost else "fail"
    return (
        f"{verdict}: cost {without_cache.estimated_cost:.6f}->{with_cache.estimated_cost:.6f} "
        f"({cost_delta:+.1%}); hit_rate {without_cache.cache_hit_rate:.4f}->"
        f"{with_cache.cache_hit_rate:.4f}; circuit_opens "
        f"{without_cache.circuit_open_count}->{with_cache.circuit_open_count}"
    )


def run_simulation(config: LabConfig, queries: list[str]) -> RunMetrics:
    """Run all named scenarios from config, or a default run if none defined."""
    if not config.scenarios:
        default_scenario = ScenarioConfig(name="default", description="baseline run")
        metrics = run_scenario(config, queries, default_scenario)
        passed, reason = evaluate_scenario(config, default_scenario, metrics)
        metrics.scenarios = {"default": f"{'pass' if passed else 'fail'}: {reason}"}
        return metrics

    combined = RunMetrics()
    for scenario in config.scenarios:
        result = run_scenario(config, queries, scenario)

        passed, reason = evaluate_scenario(config, scenario, result)
        combined.scenarios[scenario.name] = f"{'pass' if passed else 'fail'}: {reason}"

        combined.total_requests += result.total_requests
        combined.successful_requests += result.successful_requests
        combined.failed_requests += result.failed_requests
        combined.fallback_successes += result.fallback_successes
        combined.static_fallbacks += result.static_fallbacks
        combined.cache_hits += result.cache_hits
        combined.circuit_open_count += result.circuit_open_count
        combined.estimated_cost += result.estimated_cost
        combined.estimated_cost_saved += result.estimated_cost_saved
        combined.latencies_ms.extend(result.latencies_ms)
        if result.recovery_time_ms is not None:
            if combined.recovery_time_ms is None:
                combined.recovery_time_ms = result.recovery_time_ms
            else:
                combined.recovery_time_ms = (combined.recovery_time_ms + result.recovery_time_ms) / 2

    # Control experiment — reported alongside the scenarios but not folded into the
    # counters above, so the headline cache_hit_rate stays a measure of real traffic.
    combined.scenarios["cache_comparison"] = compare_cache(config, queries)

    return combined