from config import CONTEXTS
import random
import numpy as np

"""Utility functions for the snus cessation intervention simulation, including state discretization and sampling of true triggers for users."""

def discretize(value, low=0.33, high=0.66):
    if value < low:
        return "low"
    elif value < high:
        return "medium"
    else:
        return "high"

def sample_true_triggers_for_user(user):
    number_of_triggers = 1 if random.random() < 0.75 else 2

    trigger_indices = np.random.choice(
        len(CONTEXTS),
        size=number_of_triggers,
        replace=False,
        p=user.trigger_profile
    )

    return [CONTEXTS[i] for i in trigger_indices]