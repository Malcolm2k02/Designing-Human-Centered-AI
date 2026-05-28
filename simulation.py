import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


# -----------------------------
# 3. User class
# -----------------------------

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

        self.nudge_values = {nudge: 0.0 for nudge in NUDGES}
        self.nudge_counts = {nudge: 0 for nudge in NUDGES}

    def predict_risk(self, context):
        """
        Psychology-informed relapse risk prediction.
        Risk increases with addiction, stress, craving, fatigue, and social pressure.
        Risk decreases with motivation and self-efficacy.
        """

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

    def choose_nudge(self, context, risk):
        """
        Hybrid recommender:
        1. Psychology-informed rule-based recommendation
        2. Reinforcement learning personalization
        3. Option to choose no intervention
        """

        if risk < 0.30:
            preferred = "no_intervention"
        elif context == "morning_craving":
            preferred = "snus_consumption_feedback"
        elif self.strategy == "gradual_reduction":
            preferred = "small_reduction_goal"
        else:
            preferred = "economic reminder"

        if random.random() < 0.20:
            return random.choice(NUDGES)

        scores = self.nudge_values.copy()
        scores[preferred] += 0.4

        if self.fatigue > 0.6:
            scores["no_intervention"] += 0.6

        return max(scores, key=scores.get)

    def respond_to_nudge(self, risk, nudge):
        """
        Simulates response:
        skip = avoided snus
        delay = delayed use
        use = used snus
        ignore = ignored intervention
        """

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

    def update_learning(self, nudge, response):
        """
        Bandit-style reinforcement learning update.
        """

        reward = REWARD[response]

        if nudge == "no_intervention" and response == "skip":
            reward += 1

        if response == "ignore":
            reward -= self.fatigue

        self.nudge_counts[nudge] += 1
        count = self.nudge_counts[nudge]

        old_value = self.nudge_values[nudge]
        new_value = old_value + (reward - old_value) / count

        self.nudge_values[nudge] = new_value

    def update_feedback_loops(self, response, nudge):
        """
        Psychological feedback loops:
        Small successes increase self-efficacy.
        Ignored nudges increase fatigue.
        """

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

        elif response == "use":
            self.motivation = max(0.0, self.motivation - 0.005)
            self.self_efficacy = max(0.0, self.self_efficacy - 0.008)
            self.craving = min(1.0, self.craving + 0.02)

        if nudge == "economic reminder" and response in ["skip", "delay"]:
            self.stress = max(0.0, self.stress - 0.04)

        if nudge == "snus_consumption_feedback" and response in ["skip", "delay"]:
            self.stress = max(0.0, self.stress - 0.03)
        
        if nudge == "small_reduction_goal" and response in ["skip", "delay"]:
            self.stress = max(0.0, self.stress - 0.02)
        
        if nudge == "no_intervention" and response in ["skip", "delay"]:
            self.stress = max(0.0, self.stress - 0.04)

    def check_dropout(self):
        """
        Dropout becomes more likely when fatigue is high and motivation/self-efficacy are low.
        """

        dropout_probability = (
            0.005 +
            0.08 * self.fatigue +
            0.04 * (1 - self.motivation) +
            0.04 * (1 - self.self_efficacy)
        )

        dropout_probability = max(0.0, min(0.50, dropout_probability))

        if random.random() < dropout_probability:
            self.active = False


# -----------------------------
# 4. Simulation
# -----------------------------

def simulate(n_users=300, days=30, algorithm=True):
    users = []
    user_type_names = list(USER_TYPES.keys())

    for _ in range(n_users):
        user_type = random.choice(user_type_names)
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

                nudge = user.choose_nudge(context, risk)

                if nudge != "no_intervention":
                    nudges_sent += 1

                response = user.respond_to_nudge(risk, nudge)

                user.update_learning(nudge, response)
                user.update_feedback_loops(response, nudge)

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
# 5. Run simulation
# -----------------------------

baseline = simulate(n_users=300, days=30, algorithm=False)
adaptive = simulate(n_users=300, days=30, algorithm=True)


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
# 7. Plots
# -----------------------------

baseline_daily = baseline.groupby("day")["snus_used"].mean()
adaptive_daily = adaptive.groupby("day")["snus_used"].mean()

plt.figure()
plt.plot(baseline_daily.index, baseline_daily.values, label="Tracking only")
plt.plot(adaptive_daily.index, adaptive_daily.values, label="Adaptive recommender")
plt.xlabel("Day")
plt.ylabel("Average snus portions per user")
plt.title("Simulation of snus reduction over time")
plt.legend()
plt.show()


adaptive_by_type = adaptive.groupby(["day", "user_type"])["snus_used"].mean().reset_index()

plt.figure()
for user_type in adaptive_by_type["user_type"].unique():
    subset = adaptive_by_type[adaptive_by_type["user_type"] == user_type]
    plt.plot(subset["day"], subset["snus_used"], label=user_type)

plt.xlabel("Day")
plt.ylabel("Average snus portions")
plt.title("Adaptive algorithm effect by user type")
plt.legend()
plt.show()


dropout_by_type = adaptive.groupby(["day", "user_type"])["active"].mean().reset_index()

plt.figure()
for user_type in dropout_by_type["user_type"].unique():
    subset = dropout_by_type[dropout_by_type["user_type"] == user_type]
    plt.plot(subset["day"], subset["active"], label=user_type)

plt.xlabel("Day")
plt.ylabel("Proportion still active")
plt.title("Dropout over time by user type")
plt.legend()
plt.show()


money_daily = adaptive.groupby("day")["money_saved"].mean()

plt.figure()
plt.plot(money_daily.index, money_daily.values)
plt.xlabel("Day")
plt.ylabel("Average money saved per user (€)")
plt.title("Estimated money saved over time")
plt.show()


self_efficacy_daily = adaptive.groupby("day")["self_efficacy"].mean()

plt.figure()
plt.plot(self_efficacy_daily.index, self_efficacy_daily.values)
plt.xlabel("Day")
plt.ylabel("Average self-efficacy")
plt.title("Self-efficacy development over time")
plt.show()

# -----------------------------
# Cumulative money saved
# -----------------------------

cumulative_money = adaptive.groupby("day")["money_saved"].mean().cumsum()

plt.figure()
plt.plot(cumulative_money.index, cumulative_money.values)
plt.xlabel("Day")
plt.ylabel("Cumulative average money saved per user (€)")
plt.title("Cumulative money saved over time")
plt.show()