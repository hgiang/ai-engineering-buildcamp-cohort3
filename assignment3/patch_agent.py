from pydantic_ai import Agent


INPUT_COST_PER_MILLION = 0.15
OUTPUT_COST_PER_MILLION = 0.60

_original_run = Agent.run
_original_run_sync = Agent.run_sync
_patched = False
_records = []


def _record_usage(result):
    usage = result.usage()
    input_tokens = usage.input_tokens or 0
    output_tokens = usage.output_tokens or 0
    cost = (
        input_tokens * INPUT_COST_PER_MILLION
        + output_tokens * OUTPUT_COST_PER_MILLION
    ) / 1_000_000
    _records.append(
        {
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "cost": cost,
        }
    )


def patch_agent_cost_tracking():
    global _patched
    if _patched:
        return

    async def run_with_tracking(self, *args, **kwargs):
        result = await _original_run(self, *args, **kwargs)
        _record_usage(result)
        return result

    def run_sync_with_tracking(self, *args, **kwargs):
        result = _original_run_sync(self, *args, **kwargs)
        _record_usage(result)
        return result

    Agent.run = run_with_tracking
    Agent.run_sync = run_sync_with_tracking
    _patched = True


def cost_summary():
    input_tokens = sum(record["input_tokens"] for record in _records)
    output_tokens = sum(record["output_tokens"] for record in _records)
    cost = sum(record["cost"] for record in _records)
    return {
        "calls": len(_records),
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost": cost,
    }
