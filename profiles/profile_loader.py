from profiles import driving_profile
from profiles import airport_profile


def load_profile(active_profile):
    if active_profile == "driving":
        return driving_profile

    if active_profile == "airport":
        return airport_profile

    raise ValueError(f"Unknown detection profile: {active_profile}")