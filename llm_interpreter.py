def generate_warning(event, llm_enabled=False):
    """
    Generate a short human-readable warning from a structured detection event.

    This first version is intentionally local only.
    It does not call OpenAI or any external LLM provider.

    The purpose is to prove the event flow:

    deterministic detection event
        -> interpreter
        -> plain-English warning
    """

    event_type = event.get("event_type")
    detected_class = event.get("class")
    confidence = event.get("confidence")
    frame_number = event.get("frame_number")

    if not llm_enabled:
        if event_type == "unique_person_detected":
            return "Person close by. A new person was detected in the scene."

        return "Caution. A new detection event was identified."

    # Placeholder for a future real LLM integration.
    # Do not add external API calls here until key handling,
    # security, retry logic, and cost controls are ready.
    if detected_class and confidence is not None and frame_number is not None:
        return (
            f"Caution. A {detected_class} was detected "
            f"with confidence {confidence:.2f} at frame {frame_number}."
        )

    return "Caution. A detection event was identified."