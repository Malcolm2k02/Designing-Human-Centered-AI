from config import Q_TABLE, NUDGES, USER_TYPES, EPSILON_START, EPSILON_END, COST_PER_PORTION
from user import User
import random
import numpy as np
import pandas as pd
from utils import discretize, sample_true_triggers_for_user

"""File for the main simulation loop of the snus cessation intervention, including user behavior, nudge selection, and learning updates."""
def simulate(n_users=300, days=30, algorithm=True, training_mode=True):
    """Simulates the snus cessation intervention for a given number of users and days."""
    users = []

    for _ in range(n_users):
        user_type = random.choice(list(USER_TYPES.keys()))
        users.append(User(user_type))

    results = []
    event_results = []

    for day in range(1, days + 1):

        if training_mode:
            progress = (day - 1) / max(1, days - 1)
            current_epsilon = EPSILON_START + progress * (EPSILON_END - EPSILON_START)
        else:
            current_epsilon = 0.0

        for user_id, user in enumerate(users):
            user.current_day = day
            if not user.active:
                results.append({
                    "day": day,
                    "user_id": user_id,
                    "user_type": user.user_type,
                    "strategy": user.strategy,
                    "active": False,
                    "snus_used": np.nan,
                    "nudges_sent": 0,
                    "skips": 0,
                    "delays": 0,
                    "ignores": 0,
                    "money_saved": np.nan,
                    "fatigue": user.fatigue,
                    "motivation": user.motivation,
                    "self_efficacy": user.self_efficacy,
                    "inferred_trigger": user.most_likely_trigger()
                })
                continue

            daily_cravings = int(
                user.baseline_use *
                (0.8 + user.stress * 0.4 + user.addiction * 0.4 + user.craving * 0.3)
            )

            snus_used = 0
            nudges_sent = 0
            skips = 0
            delays = 0
            ignores = 0

            for _ in range(daily_cravings):

                true_triggers = sample_true_triggers_for_user(user)
                observed_trigger = user.observe_trigger(true_triggers)

                actual_risk = user.predict_risk(true_triggers)
                estimated_risk = user.estimate_risk_from_beliefs()

                actual_primary_trigger = true_triggers[0]
                inferred_trigger_before = user.most_likely_trigger()

                if not algorithm:
                    use_probability = (
                        0.35 +
                        0.35 * user.addiction +
                        0.20 * user.stress +
                        0.20 * user.craving -
                        0.20 * user.motivation -
                        0.20 * user.self_efficacy
                    )

                    use_probability = max(0.05, min(0.95, use_probability))

                    if random.random() < use_probability:
                        snus_used += 1

                    continue

                state = user.get_state(estimated_risk)

                if state not in Q_TABLE:
                    Q_TABLE[state] = {nudge: 0.0 for nudge in NUDGES}

                nudge_probability = max(
                    0.2,
                    1 - (nudges_sent * 0.20)
                )

                if random.random() < nudge_probability:
                    nudge = user.choose_nudge(estimated_risk, current_epsilon)
                else:
                    nudge = "no_intervention"

                if nudge != "no_intervention":
                    nudges_sent += 1

                response = user.respond_to_nudge(actual_risk, nudge)

                user.update_feedback_loops(response, nudge)
                user.update_trigger_beliefs(observed_trigger, response)

                event_results.append({
                    "day": day,
                    "user_id": user_id,
                    "user_type": user.user_type,
                    "strategy": user.strategy,
                    "actual_triggers": ",".join(true_triggers),
                    "actual_primary_trigger": actual_primary_trigger,
                    "observed_trigger": observed_trigger,
                    "inferred_trigger": inferred_trigger_before,
                    "nudge": nudge,
                    "response": response,
                    "actual_risk": actual_risk,
                    "estimated_risk": estimated_risk,
                    "fatigue": user.fatigue,
                    "motivation": user.motivation,
                    "self_efficacy": user.self_efficacy
                })

                next_estimated_risk = user.estimate_risk_from_beliefs()
                next_state = user.get_state(next_estimated_risk)

                if training_mode:
                    user.update_learning(state, nudge, response, next_state)

                if response == "skip":
                    skips += 1
                elif response == "delay":
                    delays += 1
                    snus_used += 1
                elif response == "ignore":
                    ignores += 1
                    snus_used += 1
                elif response == "use":
                    snus_used += 1

                if nudges_sent > 4:
                    user.fatigue = min(1.0, user.fatigue + 0.005)

            user.check_dropout()

            money_saved = max(0, (user.baseline_use - snus_used) * COST_PER_PORTION)

            beliefs = sorted(
                user.trigger_beliefs.items(),
                key=lambda x: x[1],
                reverse=True
            )

            results.append({
                "day": day,
                "user_id": user_id,
                "user_type": user.user_type,
                "strategy": user.strategy,
                "active": user.active,
                "snus_used": snus_used,
                "nudges_sent": nudges_sent,
                "skips": skips,
                "delays": delays,
                "ignores": ignores,
                "money_saved": money_saved,
                "fatigue": user.fatigue,
                "motivation": user.motivation,
                "self_efficacy": user.self_efficacy,

                # Most likely inferred trigger
                "inferred_trigger": beliefs[0][0],

                # Second most likely inferred trigger
                "secondary_trigger": beliefs[1][0],

                # Probability/confidence of top trigger
                "trigger_confidence": beliefs[0][1]
            })
    return pd.DataFrame(results), pd.DataFrame(event_results)
