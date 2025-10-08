import os

def os_available_cpus() -> int:
    # Respect CPU affinity if set (e.g., taskset / cgroups)
    try:
        return len(os.sched_getaffinity(0))
    except Exception:
        return os.cpu_count() or 1

def available_cpus() -> int:
    """How many CPUs can we really use right now? Prefer Ray's view if initialized."""
    try:
        import ray
        if ray.is_initialized():
            # Ray reports fractional CPUs as floats; floor to int
            return int(ray.available_resources().get("CPU", 0)) or 0
    except Exception:
        pass
    return os_available_cpus()

def recommend_actor_count(requested_actors: int, per_actor_cpus: int = 4, reserve_cpus: int = 1) -> int:
    """
    Cap the number of Ray actors so total reserved CPUs do not exceed what's available.
    reserve_cpus: keep a small buffer for the driver / OS.
    """
    avail = available_cpus()
    # if Ray isn't initialized yet, avail=OS cpus; otherwise Ray's free CPUs
    usable = max(0, avail - max(0, reserve_cpus))
    max_by_cpu = max(1, usable // max(1, per_actor_cpus))
    return max(1, min(requested_actors, max_by_cpu))
