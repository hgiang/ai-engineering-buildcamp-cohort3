from patch_agent import cost_summary, patch_agent_cost_tracking


def pytest_configure(config):
    patch_agent_cost_tracking()


def pytest_terminal_summary(terminalreporter, exitstatus, config):
    summary = cost_summary()
    terminalreporter.write_line(
        "LLM usage: "
        f"{summary['calls']} calls, "
        f"{summary['input_tokens']} input tokens, "
        f"{summary['output_tokens']} output tokens, "
        f"${summary['cost']:.6f}"
    )
