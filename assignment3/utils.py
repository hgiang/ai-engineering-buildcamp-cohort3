def collect_tools(messages):
    tools = []
    for message in messages:
        for part in getattr(message, "parts", []):
            if getattr(part, "part_kind", None) == "tool-call":
                tools.append(part.tool_name)
    return tools
