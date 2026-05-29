import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)
# a simplified partially observable contextual reinforcement learning 
# recommender system for nicotine intervention.

# -----------------------------
# 1. Intervention actions
# -----------------------------

NUDGES = [
    "no_intervention",
    "economic reminder",
    "snus_consumption_feedback",
    "small_reduction_goal"
]

REWARD = {
    "skip": 3,
    "delay": 1.5,
    "use": -0.5,
    "ignore": -1
}

COST_PER_PORTION = 0.23 # Average cost of a snus portion in euros

ALPHA = 0.1 # Learning rate for Q-learning updates
GAMMA = 0.9 # Discount factor for future rewards in Q-learning updates

EPSILON_START = 0.30
EPSILON_END = 0.05

Q_TABLE = {}


# -----------------------------
# 2. Contexts / hidden triggers
# -----------------------------

CONTEXTS = [
    "after_meal",
    "social_setting",
    "stress",
    "studying",
    "alcohol_context",
    "morning_craving",
    "boredom",
    "sleeping"
]

CONTEXT_RISK = {
    "after_meal": 0.15,
    "social_setting": 0.20,
    "stress": 0.25,
    "studying": 0.10,
    "alcohol_context": 0.25,
    "morning_craving": 0.20,
    "boredom": 0.10,
    "sleeping": 0.18
}

OBSERVATION_MODEL = {
    "after_meal": {
        "after_meal": 0.60, "boredom": 0.15, "social_setting": 0.10,
        "stress": 0.05, "studying": 0.03, "morning_craving": 0.03,
        "alcohol_context": 0.02, "sleeping": 0.02
    },
    "social_setting": {
        "social_setting": 0.55, "alcohol_context": 0.20, "stress": 0.10,
        "after_meal": 0.05, "boredom": 0.05, "studying": 0.02,
        "morning_craving": 0.02, "sleeping": 0.01
    },
    "stress": {
        "stress": 0.55, "studying": 0.15, "boredom": 0.10,
        "social_setting": 0.08, "after_meal": 0.04, "sleeping": 0.03,
        "morning_craving": 0.03, "alcohol_context": 0.02
    },
    "studying": {
        "studying": 0.60, "stress": 0.20, "boredom": 0.08,
        "after_meal": 0.04, "social_setting": 0.03, "morning_craving": 0.02,
        "sleeping": 0.02, "alcohol_context": 0.01
    },
    "alcohol_context": {
        "alcohol_context": 0.65, "social_setting": 0.20, "stress": 0.05,
        "boredom": 0.04, "after_meal": 0.03, "morning_craving": 0.01,
        "studying": 0.01, "sleeping": 0.01
    },
    "morning_craving": {
        "morning_craving": 0.65, "stress": 0.10, "sleeping": 0.10,
        "after_meal": 0.05, "boredom": 0.04, "studying": 0.03,
        "social_setting": 0.02, "alcohol_context": 0.01
    },
    "boredom": {
        "boredom": 0.55, "stress": 0.15, "studying": 0.10,
        "social_setting": 0.08, "after_meal": 0.05, "sleeping": 0.03,
        "morning_craving": 0.02, "alcohol_context": 0.02
    },
    "sleeping": {
        "sleeping": 0.60, "morning_craving": 0.15, "stress": 0.10,
        "boredom": 0.05, "after_meal": 0.04, "studying": 0.03,
        "social_setting": 0.02, "alcohol_context": 0.01
    }
}


# -----------------------------
# 3. Psychology-informed user groups
# -----------------------------

USER_TYPES = {
    "High relapse risk / high intake": {
        "baseline_use": (10, 16),
        "motivation": (0.3, 0.6),
        "addiction": (0.8, 1.0),
        "stress": (0.6, 1.0),
        "adherence": (0.3, 0.6),
        "self_efficacy": (0.2, 0.5)
    },
    "High relapse risk / low intake": {
        "baseline_use": (3, 7),
        "motivation": (0.3, 0.7),
        "addiction": (0.6, 0.9),
        "stress": (0.7, 1.0),
        "adherence": (0.4, 0.7),
        "self_efficacy": (0.2, 0.6)
    },
    "Low relapse risk / high intake": {
        "baseline_use": (8, 13),
        "motivation": (0.7, 1.0),
        "addiction": (0.5, 0.8),
        "stress": (0.2, 0.6),
        "adherence": (0.6, 0.9),
        "self_efficacy": (0.6, 0.9)
    },
    "Low relapse risk / low intake": {
        "baseline_use": (2, 6),
        "motivation": (0.7, 1.0),
        "addiction": (0.2, 0.5),
        "stress": (0.2, 0.5),
        "adherence": (0.7, 1.0),
        "self_efficacy": (0.6, 1.0)
    }
}

class User:
    def __init__(self, user_type):
        profile = USER_TYPES[user_type]

        self.user_type = user_type
        self.baseline_use = random.randint(*profile["baseline_use"])
        self.motivation = random.uniform(*profile["motivation"])
        self.addiction = random.uniform(*profile["addiction"])
        self.stress = random.uniform(*profile["stress"])
        self.adherence = random.uniform(*profile["adherence"])
        self.self_efficacy = random.uniform(*profile["self_efficacy"])

        self.craving = random.uniform(0.3, 1.0)
        self.social_pressure = random.uniform(0.0, 1.0)

        self.fatigue = 0.0
        self.active = True

        self.strategy = random.choice(["cold_turkey", "gradual_reduction"])

        self.current_day = 1
        self.success_streak = 0
        self.total_successes = 0

        self.trigger_beliefs = {
            trigger: 1 / len(CONTEXTS) for trigger in CONTEXTS
        }

        self.trigger_profile = np.random.dirichlet(np.ones(len(CONTEXTS)))

    def most_likely_trigger(self):
        return max(self.trigger_beliefs, key=self.trigger_beliefs.get)

    def get_state(self, estimated_risk):
        return (
            self.user_type,
            self.most_likely_trigger(),
            discretize(estimated_risk),
            discretize(self.fatigue),
            self.strategy
        )

    def observe_trigger(self, true_triggers):
        """
        Structured noisy observation model.

        The app does not observe the true trigger directly.
        It observes a noisy clue that can confuse related triggers.
        """

        true_trigger = random.choice(true_triggers)

        possible_observations = list(OBSERVATION_MODEL[true_trigger].keys())
        probabilities = list(OBSERVATION_MODEL[true_trigger].values())

        return np.random.choice(possible_observations, p=probabilities)

    def update_trigger_beliefs(self, observed_trigger, response):
        if response == "use":
            self.trigger_beliefs[observed_trigger] += 0.20
        elif response == "ignore":
            self.trigger_beliefs[observed_trigger] += 0.12
        elif response == "delay":
            self.trigger_beliefs[observed_trigger] += 0.05
        elif response == "skip":
            self.trigger_beliefs[observed_trigger] += 0.02

        for trigger in self.trigger_beliefs:
            if trigger != observed_trigger:
                self.trigger_beliefs[trigger] *= 0.98

        total = sum(self.trigger_beliefs.values())

        for trigger in self.trigger_beliefs:
            self.trigger_beliefs[trigger] /= total

    def predict_risk(self, true_triggers):
        trigger_risk = sum(CONTEXT_RISK[t] for t in true_triggers)

        risk = (
            0.25 * self.addiction +
            0.20 * self.stress +
            0.20 * self.craving +
            0.15 * self.fatigue +
            0.10 * self.social_pressure -
            0.20 * self.motivation -
            0.20 * self.self_efficacy +
            trigger_risk
        )

        # Cold turkey is harder early, but can become more effective later.
        if self.strategy == "cold_turkey":
            if self.current_day <= 10:
                risk += 0.08
                risk += random.uniform(-0.08, 0.08)  # higher early variance
            else:
                risk -= min(0.08, self.total_successes * 0.003)

        # Gradual reduction is smoother and easier early, but improves more slowly.
        elif self.strategy == "gradual_reduction":
            if self.current_day <= 10:
                risk -= 0.04
            else:
                risk -= min(0.04, self.total_successes * 0.0015)

        return max(0.0, min(1.0, risk))

    def estimate_risk_from_beliefs(self):
        expected_trigger_risk = 0.0

        for trigger, probability in self.trigger_beliefs.items():
            expected_trigger_risk += probability * CONTEXT_RISK[trigger]

        risk = (
            0.25 * self.addiction +
            0.20 * self.stress +
            0.20 * self.craving +
            0.15 * self.fatigue +
            0.10 * self.social_pressure -
            0.20 * self.motivation -
            0.20 * self.self_efficacy +
            expected_trigger_risk
        )

        if self.strategy == "cold_turkey":
            if self.current_day <= 10:
                risk += 0.06
            else:
                risk -= min(0.06, self.total_successes * 0.0025)

        elif self.strategy == "gradual_reduction":
            if self.current_day <= 10:
                risk -= 0.03
            else:
                risk -= min(0.035, self.total_successes * 0.0012)

        return max(0.0, min(1.0, risk))

    def choose_nudge(self, estimated_risk, epsilon):
        state = self.get_state(estimated_risk)

        if state not in Q_TABLE:
            Q_TABLE[state] = {nudge: 0.0 for nudge in NUDGES}

        if random.random() < epsilon:
            return random.choice(NUDGES)

        return max(Q_TABLE[state], key=Q_TABLE[state].get)

    def respond_to_nudge(self, actual_risk, nudge):
        if nudge == "no_intervention":
            use_probability = (
                0.30 +
                0.35 * self.addiction +
                0.25 * self.craving +
                0.15 * self.stress -
                0.20 * self.self_efficacy
            )
            use_probability = max(0.05, min(0.95, use_probability))

            if random.random() < use_probability:
                return "use"
            else:
                return "skip"

        engage_probability = self.adherence - self.fatigue
        engage_probability = max(0.05, min(0.95, engage_probability))

        if random.random() > engage_probability:
            return "ignore"

        nudge_quality = {
            "economic reminder": 0.65,
            "snus_consumption_feedback": 0.70,
            "small_reduction_goal": 0.55,
            "no_intervention": 0.00
        }

        success_probability = (
            nudge_quality[nudge]
            * self.motivation
            * self.self_efficacy
            * (1 - self.addiction * 0.4)
        )

        if self.strategy == "cold_turkey":
            if self.current_day <= 10:
                success_probability -= 0.06
            else:
                success_probability += min(0.08, self.total_successes * 0.002)

        elif self.strategy == "gradual_reduction":
            if self.current_day <= 10:
                success_probability += 0.04
            else:
                success_probability += min(0.035, self.total_successes * 0.001)

        success_probability -= actual_risk * 0.15
        success_probability = max(0.05, min(0.90, success_probability))

        r = random.random()

        if r < success_probability:
            return "skip"
        elif r < success_probability + 0.25:
            return "delay"
        else:
            return "use"

    def update_learning(self, state, nudge, response, next_state):
        reward = REWARD[response]

        if nudge == "no_intervention":
            if response == "skip":
                reward = 0.5
            elif response == "use":
                reward = -1.5

        if response == "ignore":
            reward -= self.fatigue * 0.3

        if state not in Q_TABLE:
            Q_TABLE[state] = {n: 0.0 for n in NUDGES}

        if next_state not in Q_TABLE:
            Q_TABLE[next_state] = {n: 0.0 for n in NUDGES}

        old_q = Q_TABLE[state][nudge]
        best_next_q = max(Q_TABLE[next_state].values())

        new_q = old_q + ALPHA * (reward + GAMMA * best_next_q - old_q)

        Q_TABLE[state][nudge] = new_q

    def update_feedback_loops(self, response, nudge):
        if response in ["skip", "delay"]:
            self.success_streak += 1
            self.total_successes += 1
        else:
            self.success_streak = 0

        if response == "skip":
            self.motivation = min(1.0, self.motivation + 0.010)
            self.self_efficacy = min(1.0, self.self_efficacy + 0.010)
            self.craving = max(0.0, self.craving - 0.025)
            self.fatigue = max(0.0, self.fatigue - 0.008)

            if self.strategy == "cold_turkey":
                self.self_efficacy = min(1.0, self.self_efficacy + 0.006)
            elif self.strategy == "gradual_reduction":
                self.motivation = min(1.0, self.motivation + 0.003)

        elif response == "delay":
            self.motivation = min(1.0, self.motivation + 0.005)
            self.self_efficacy = min(1.0, self.self_efficacy + 0.006)
            self.craving = max(0.0, self.craving - 0.012)
            self.fatigue = max(0.0, self.fatigue - 0.004)

        elif response == "ignore":
            self.fatigue = min(1.0, self.fatigue + 0.035)
            self.motivation = max(0.0, self.motivation - 0.002)

        elif response == "use":
            self.motivation = max(0.0, self.motivation - 0.004)
            self.self_efficacy = max(0.0, self.self_efficacy - 0.006)
            self.craving = min(1.0, self.craving + 0.015)

            if self.strategy == "cold_turkey":
                self.self_efficacy = max(0.0, self.self_efficacy - 0.006)
                self.craving = min(1.0, self.craving + 0.006)
            elif self.strategy == "gradual_reduction":
                self.self_efficacy = max(0.0, self.self_efficacy - 0.002)

        if response in ["skip", "delay"]:
            if nudge == "economic reminder":
                self.motivation = min(1.0, self.motivation + 0.006)

            elif nudge == "snus_consumption_feedback":
                self.self_efficacy = min(1.0, self.self_efficacy + 0.006)
                self.craving = max(0.0, self.craving - 0.006)

            elif nudge == "small_reduction_goal":
                self.self_efficacy = min(1.0, self.self_efficacy + 0.008)
                self.motivation = min(1.0, self.motivation + 0.003)

    def check_dropout(self):
        dropout_probability = (
            0.002 +
            0.03 * self.fatigue +
            0.02 * (1 - self.motivation) +
            0.02 * (1 - self.self_efficacy)
        )

        dropout_probability = max(0.0, min(0.12, dropout_probability))

        if random.random() < dropout_probability:
            self.active = False


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

def simulate(n_users=300, days=30, algorithm=True, training_mode=True):
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


# -----------------------------
# Train and evaluate
# -----------------------------

Q_TABLE = {}

training, training_events = simulate(
    n_users=5000,
    days=30,
    algorithm=True,
    training_mode=True
)

adaptive, adaptive_events = simulate(
    n_users=1000,
    days=30,
    algorithm=True,
    training_mode=False
)

baseline, baseline_events = simulate(
    n_users=1000,
    days=30,
    algorithm=False,
    training_mode=False
)

# -----------------------------
# 6. Evaluation
# -----------------------------

def summarize(df, name):
    active_df = df[df["active"] == True]

    avg_day_1 = active_df[active_df["day"] == 1]["snus_used"].mean()
    avg_day_30 = active_df[active_df["day"] == 30]["snus_used"].mean()

    reduction = ((avg_day_1 - avg_day_30) / avg_day_1) * 100

    dropout_rate = 1 - (df[df["day"] == 30]["active"].mean())

    total_money_saved = active_df["money_saved"].sum()
    avg_money_saved = active_df.groupby("user_id")["money_saved"].sum().mean()

    print(f"\n{name}")
    print("-" * 50)
    print(f"Average snus use day 1:       {avg_day_1:.2f}")
    print(f"Average snus use day 30:      {avg_day_30:.2f}")
    print(f"Reduction:                    {reduction:.1f}%")
    print(f"Dropout rate:                 {dropout_rate:.1%}")
    print(f"Total estimated money saved:  €{total_money_saved:.2f}")
    print(f"Avg money saved per user:     €{avg_money_saved:.2f}")

    print(f"Average nudges per user/day:  {active_df['nudges_sent'].mean():.2f}")
    print(f"Average skips per user/day:   {active_df['skips'].mean():.2f}")
    print(f"Average delays per user/day:  {active_df['delays'].mean():.2f}")
    print(f"Average ignores per user/day: {active_df['ignores'].mean():.2f}")
    print(f"Average fatigue day 30:       {active_df[active_df['day'] == 30]['fatigue'].mean():.2f}")
    print(f"Average self-efficacy day 30: {active_df[active_df['day'] == 30]['self_efficacy'].mean():.2f}")


summarize(baseline, "Tracking-only baseline")
summarize(adaptive, "Psychology-informed adaptive recommender")


# -----------------------------
# 7. Improved plots
# -----------------------------

plt.rcParams.update({
    "figure.figsize": (9, 5),
    "axes.grid": True,
    "grid.alpha": 0.3,
    "font.size": 11
})

# Print some learned Q-values for interpretability
# -----------------------------
# Inspect learned Q-table
# -----------------------------

import random

print("\nRandom learned Q-values:\n")

# Only keep states that were actually trained
# (at least one Q-value changed from zero)
trained_states = [
    state for state, actions in Q_TABLE.items()
    if any(value != 0 for value in actions.values())
]

# Randomly sample up to 5 trained states
random_states = random.sample(
    trained_states,
    min(5, len(trained_states))
)

for state in random_states:

    actions = Q_TABLE[state]

    # Find the best learned action
    best_action = max(actions, key=actions.get)

    print("STATE:")
    print(f"""
        User type:        {state[0]}
        Inferred trigger: {state[1]}
        Risk level:       {state[2]}
        Fatigue level:    {state[3]}
        Strategy:         {state[4]}
        """)

    print("\nQ-values:")

    for action, value in actions.items():
        print(f"  {action}: {value:.2f}")

    print(f"\nBest learned action: {best_action}")

    print("\n" + "-" * 60 + "\n")

trigger_accuracy = (
    adaptive_events["actual_primary_trigger"] ==
    adaptive_events["inferred_trigger"]
).mean()

print(f"Event-level trigger inference accuracy: {trigger_accuracy:.1%}")

# Plot 1: Baseline vs adaptive recommender
baseline_active = baseline[baseline["active"] == True]
adaptive_active = adaptive[adaptive["active"] == True]

baseline_daily = baseline_active.groupby("day")["snus_used"].mean()
adaptive_daily = adaptive_active.groupby("day")["snus_used"].mean()

plt.figure()
plt.plot(baseline_daily.index, baseline_daily.values, linewidth=2.5, label="Tracking-only baseline")
plt.plot(adaptive_daily.index, adaptive_daily.values, linewidth=2.5, label="Adaptive recommender")

plt.xlabel("Simulation day")
plt.ylabel("Average snus portions used per active user")
plt.title("Average Daily Snus Use: Baseline vs Adaptive Recommender")
plt.legend()
plt.tight_layout()
plt.show()


# Plot 2: Adaptive recommender by user type
adaptive_active_by_type = (
    adaptive_active
    .groupby(["day", "user_type"])["snus_used"]
    .mean()
    .reset_index()
)

plt.figure()
for user_type in adaptive_active_by_type["user_type"].unique():
    subset = adaptive_active_by_type[adaptive_active_by_type["user_type"] == user_type]
    plt.plot(subset["day"], subset["snus_used"], linewidth=2, label=user_type)

plt.xlabel("Simulation day")
plt.ylabel("Average snus portions used per active user")
plt.title("Adaptive Recommender Effect by User Type")
plt.legend(title="User type", fontsize=9)
plt.tight_layout()
plt.show()


# Plot 3: Dropout / retention over time by user type
dropout_by_type = (
    adaptive
    .groupby(["day", "user_type"])["active"]
    .mean()
    .reset_index()
)

plt.figure()
for user_type in dropout_by_type["user_type"].unique():
    subset = dropout_by_type[dropout_by_type["user_type"] == user_type]
    plt.plot(subset["day"], subset["active"], linewidth=2, label=user_type)

plt.xlabel("Simulation day")
plt.ylabel("Proportion of users still active")
plt.title("User Retention Over Time by User Type")
plt.ylim(0, 1.05)
plt.legend(title="User type", fontsize=9)
plt.tight_layout()
plt.show()


# Plot 4: Daily money saved by user type
money_daily_by_type = (
    adaptive_active
    .groupby(["day", "user_type"])["money_saved"]
    .mean()
    .reset_index()
)

plt.figure()
for user_type in money_daily_by_type["user_type"].unique():
    subset = money_daily_by_type[money_daily_by_type["user_type"] == user_type]
    plt.plot(subset["day"], subset["money_saved"], linewidth=2, label=user_type)

plt.xlabel("Simulation day")
plt.ylabel("Average daily money saved per active user (€)")
plt.title("Daily Estimated Money Saved by User Type")
plt.legend(title="User type", fontsize=9)
plt.tight_layout()
plt.show()


# Plot 5: Cumulative money saved by user type
money_daily_by_type["cumulative_money_saved"] = (
    money_daily_by_type
    .groupby("user_type")["money_saved"]
    .cumsum()
)

plt.figure()
for user_type in money_daily_by_type["user_type"].unique():
    subset = money_daily_by_type[money_daily_by_type["user_type"] == user_type]
    plt.plot(
        subset["day"],
        subset["cumulative_money_saved"],
        linewidth=2,
        label=user_type
    )

plt.xlabel("Simulation day")
plt.ylabel("Cumulative average money saved per active user (€)")
plt.title("Cumulative Estimated Money Saved by User Type")
plt.legend(title="User type", fontsize=9)
plt.tight_layout()
plt.show()

# Plot 6: Most commonly inferred triggers
trigger_counts = (
    adaptive_active["inferred_trigger"]
    .value_counts()
)

plt.figure()

plt.bar(
    trigger_counts.index,
    trigger_counts.values
)

plt.xticks(rotation=25)
plt.xlabel("Most likely inferred trigger")
plt.ylabel("Number of active users")
plt.title("Distribution of Inferred User Triggers")

plt.tight_layout()
plt.show()

# Plot 7: Average snus use by inferred trigger
# Plot 7: Average snus use by inferred trigger
trigger_effect = (
    adaptive_active
    .groupby(["day", "inferred_trigger"])["snus_used"]
    .mean()
    .reset_index()
)

plt.figure()

for trigger in trigger_effect["inferred_trigger"].unique():

    subset = trigger_effect[
        trigger_effect["inferred_trigger"] == trigger
    ].copy()

    # Smooth only the numeric snus_used column
    subset["snus_used_smoothed"] = (
        subset["snus_used"]
        .rolling(window=3, min_periods=1)
        .mean()
    )

    plt.plot(
        subset["day"],
        subset["snus_used_smoothed"],
        linewidth=2,
        label=trigger
    )

plt.xlabel("Simulation day")
plt.ylabel("Average snus portions")
plt.title("Adaptive Recommender Effect by Inferred Trigger")

plt.legend(fontsize=9)
plt.tight_layout()
plt.show()

confusion = pd.crosstab(
    adaptive_events["actual_primary_trigger"],
    adaptive_events["inferred_trigger"],
    normalize="index"
)

plt.figure(figsize=(9, 6))
plt.imshow(confusion, aspect="auto")

plt.xticks(
    range(len(confusion.columns)),
    confusion.columns,
    rotation=45,
    ha="right"
)

plt.yticks(
    range(len(confusion.index)),
    confusion.index
)

plt.xlabel("Inferred trigger at craving moment")
plt.ylabel("Actual primary trigger")
plt.title("Event-Level Trigger Inference Accuracy")

plt.colorbar(label="Proportion")
plt.tight_layout()
plt.show()

fatigue_by_type = (
    adaptive.groupby(["day", "user_type"])["fatigue"]
    .mean()
    .reset_index()
)

plt.figure(figsize=(10, 6))

for user_type in fatigue_by_type["user_type"].unique():

    subset = fatigue_by_type[
        fatigue_by_type["user_type"] == user_type
    ]

    plt.plot(
        subset["day"],
        subset["fatigue"],
        linewidth=2,
        label=user_type
    )

plt.xlabel("Simulation day")
plt.ylabel("Average fatigue")
plt.title("Intervention Fatigue by User Type")

plt.legend()
plt.tight_layout()
plt.show()