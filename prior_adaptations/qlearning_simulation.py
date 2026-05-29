import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

random.seed(42)
np.random.seed(42)


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
    "use": -1,
    "ignore": -2
}

COST_PER_PORTION = 0.23

ALPHA = 0.1
GAMMA = 0.9

EPSILON_START = 0.30
EPSILON_END = 0.05

Q_TABLE = {}


# -----------------------------
# 2. Psychology-informed user groups
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

    def get_state(self, context, risk):
        return (
            self.user_type,
            context,
            discretize(risk),
            discretize(self.fatigue),
            self.strategy
        )

    def predict_risk(self, context):
        context_risk = {
            "after_meal": 0.15,
            "social_setting": 0.20,
            "stress": 0.25,
            "studying": 0.10,
            "alcohol_context": 0.25,
            "morning_craving": 0.20,
            "boredom": 0.10
        }

        risk = (
            0.25 * self.addiction +
            0.20 * self.stress +
            0.20 * self.craving +
            0.15 * self.fatigue +
            0.10 * self.social_pressure -
            0.20 * self.motivation -
            0.20 * self.self_efficacy +
            context_risk.get(context, 0.0)
        )

        return max(0.0, min(1.0, risk))

    def choose_nudge(self, context, risk, epsilon):
        state = self.get_state(context, risk)

        if state not in Q_TABLE:
            Q_TABLE[state] = {nudge: 0.0 for nudge in NUDGES}

        if random.random() < epsilon:
            return random.choice(NUDGES)

        return max(Q_TABLE[state], key=Q_TABLE[state].get)

    def respond_to_nudge(self, risk, nudge):
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

        success_probability -= risk * 0.15
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
                reward = -2

        if response == "ignore":
            reward -= self.fatigue

        if next_state not in Q_TABLE:
            Q_TABLE[next_state] = {n: 0.0 for n in NUDGES}

        old_q = Q_TABLE[state][nudge]
        best_next_q = max(Q_TABLE[next_state].values())

        new_q = old_q + ALPHA * (reward + GAMMA * best_next_q - old_q)

        Q_TABLE[state][nudge] = new_q

    def update_feedback_loops(self, response, nudge):
        if response == "skip":
            self.motivation = min(1.0, self.motivation + 0.015)
            self.self_efficacy = min(1.0, self.self_efficacy + 0.025)
            self.craving = max(0.0, self.craving - 0.03)
            self.fatigue = max(0.0, self.fatigue - 0.01)

        elif response == "delay":
            self.motivation = min(1.0, self.motivation + 0.008)
            self.self_efficacy = min(1.0, self.self_efficacy + 0.012)
            self.craving = max(0.0, self.craving - 0.015)
            self.fatigue = max(0.0, self.fatigue - 0.005)

        elif response == "ignore":
            self.fatigue = min(1.0, self.fatigue + 0.05)
            self.motivation = max(0.0, self.motivation - 0.003)

        elif response == "use":
            self.motivation = max(0.0, self.motivation - 0.005)
            self.self_efficacy = max(0.0, self.self_efficacy - 0.008)
            self.craving = min(1.0, self.craving + 0.02)

        if response in ["skip", "delay"]:
            if nudge == "economic reminder":
                self.motivation = min(1.0, self.motivation + 0.010)

            elif nudge == "snus_consumption_feedback":
                self.self_efficacy = min(1.0, self.self_efficacy + 0.010)
                self.craving = max(0.0, self.craving - 0.010)

            elif nudge == "small_reduction_goal":
                self.self_efficacy = min(1.0, self.self_efficacy + 0.015)
                self.motivation = min(1.0, self.motivation + 0.005)

    def check_dropout(self):
        dropout_probability = (
            0.005 +
            0.08 * self.fatigue +
            0.04 * (1 - self.motivation) +
            0.04 * (1 - self.self_efficacy)
        )

        dropout_probability = max(0.0, min(0.50, dropout_probability))

        if random.random() < dropout_probability:
            self.active = False


def discretize(value, low=0.33, high=0.66):
    if value < low:
        return "low"
    elif value < high:
        return "medium"
    else:
        return "high"


def simulate(n_users=300, days=30, algorithm=True, training_mode=True):
    users = []

    for _ in range(n_users):
        user_type = random.choice(list(USER_TYPES.keys()))
        users.append(User(user_type))

    results = []

    contexts = [
        "after_meal",
        "social_setting",
        "stress",
        "studying",
        "alcohol_context",
        "morning_craving",
        "boredom"
    ]

    for day in range(1, days + 1):

        if training_mode:
            progress = (day - 1) / max(1, days - 1)
            current_epsilon = EPSILON_START + progress * (EPSILON_END - EPSILON_START)
        else:
            current_epsilon = 0.0

        for user_id, user in enumerate(users):

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
                    "self_efficacy": user.self_efficacy
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
                context = random.choice(contexts)
                risk = user.predict_risk(context)

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

                state = user.get_state(context, risk)
                nudge = user.choose_nudge(context, risk, current_epsilon)

                if nudge != "no_intervention":
                    nudges_sent += 1

                response = user.respond_to_nudge(risk, nudge)

                user.update_feedback_loops(response, nudge)

                next_risk = user.predict_risk(context)
                next_state = user.get_state(context, next_risk)

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
                    user.fatigue = min(1.0, user.fatigue + 0.02)

            user.check_dropout()

            money_saved = max(0, (user.baseline_use - snus_used) * COST_PER_PORTION)

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
                "self_efficacy": user.self_efficacy
            })

    return pd.DataFrame(results)


# -----------------------------
# Train and evaluate
# -----------------------------

Q_TABLE = {}

training = simulate(
    n_users=5000,
    days=30,
    algorithm=True,
    training_mode=True
)

adaptive = simulate(
    n_users=1000,
    days=30,
    algorithm=True,
    training_mode=False
)

baseline = simulate(
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

# Randomly sample 5 states from the Q-table
random_states = random.sample(list(Q_TABLE.keys()), min(5, len(Q_TABLE)))

for state in random_states:

    actions = Q_TABLE[state]

    # Find action with highest learned value
    best_action = max(actions, key=actions.get)

    print("STATE:")
    print(state)

    print("\nQ-values:")
    for action, value in actions.items():
        print(f"  {action}: {value:.2f}")

    print(f"\nBest learned action: {best_action}")

    print("\n" + "-" * 60 + "\n")

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