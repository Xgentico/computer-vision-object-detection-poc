from datetime import datetime


# =========================
# IN-MEMORY WARNING STATE
# =========================

MAX_WARNINGS_TO_KEEP = 100

_warning_events = []


def add_warning_event(warning_event):
    """
    Store a warning event in memory so the browser dashboard can display it.

    This is intentionally in-memory for now.
    The permanent record is still saved in the run summary JSON.
    """

    event_to_store = warning_event.copy()

    event_to_store["created_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    _warning_events.append(event_to_store)

    if len(_warning_events) > MAX_WARNINGS_TO_KEEP:
        del _warning_events[0]


def get_warning_events():
    """
    Return the current in-memory warning events.
    """

    return {
        "status": "ok",
        "warning_count": len(_warning_events),
        "warnings": _warning_events
    }


def clear_warning_events():
    """
    Clear current in-memory warning events.
    This does not delete JSON run summaries.
    """

    _warning_events.clear()

    return {
        "status": "ok",
        "message": "Warning events cleared.",
        "warning_count": 0,
        "warnings": []
    }