import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# -----------------------------
# 1. Nudge options / interventions
# -----------------------------

NUDGES = [
    "delay_challenge",
    "breathing_exercise",
    "walk",
    "drink_water",
    "ai_coach_message",
    "social_support",
    "educational_reminder"
]

REWARD = {
    "skip": 2,
    "delay": 1,
    "use": 0,
    "ignore": -1
}


# -----------------------------
# 2. User profiles
# -----------------------------

USER_TYPES = {
    "Motivated reducer": {
        "baseline_use": (5, 9),
        "motivation": (0.7, 1.0),
        "addiction": (0.3, 0.6),
        "stress": (0.2, 0.5),
        "adherence": (0.7, 1.0)
    },
    "Stressed student": {
        "baseline_use": (6, 11),
        "motivation": (0.4, 0.8),
        "addiction": (0.4, 0.8),
        "stress": (0.7, 1.0),
        "adherence": (0.4, 0.8)
    },
    "Heavy dependent user": {
        "baseline_use": (10, 16),
        "motivation": (0.3, 0.7),
        "addiction": (0.8, 1.0),
        "stress": (0.5, 0.9),
        "adherence": (0.3, 0.7)
    },
    "Casual user": {
        "baseline_use": (2, 6),
        "motivation": (0.5, 0.9),
        "addiction": (0.2, 0.5),
        "stress": (0.2, 0.6),
        "adherence": (0.5, 0.9)
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

        self.fatigue = 0.0
        self.active = True

        # Estimated value of each nudge.
        # This is the simple reinforcement learning / bandit part.
        self.nudge_values = {nudge: 0.0 for nudge in NUDGES}
        self.nudge_counts = {nudge: 0 for nudge in NUDGES}

    def predict_risk(self):
        """
        Risk prediction:
        Higher addiction, stress, and fatigue increase risk.
        Higher motivation lowers risk.
        """
        risk = (
            0.45 * self.addiction +
            0.35 * self.stress +
            0.20 * self.fatigue -
            0.30 * self.motivation
        )

        return max(0.0, min(1.0, risk))

    def choose_nudge(self, context):
        """
        Hybrid recommender:
        1. Knowledge-based rule
        2. Reinforcement learning/bandit personalization
        """

        # Knowledge-based recommendation
        if context == "stress":
            preferred = "breathing_exercise"
        elif context == "boredom":
            preferred = "walk"
        elif context == "after_meal":
            preferred = "delay_challenge"
        elif context == "social":
            preferred = "social_support"
        else:
            preferred = "ai_coach_message"

        # Exploration: sometimes try random nudges
        if random.random() < 0.20:
            return random.choice(NUDGES)

        # Exploitation: choose best known nudge, but boost knowledge-based suggestion
        scores = self.nudge_values.copy()
        scores[preferred] += 0.3

        return max(scores, key=scores.get)

    def respond_to_nudge(self, risk, nudge):
        """
        Simulates user response:
        skip / delay / use / ignore
        """

        # Probability that the user engages with the nudge
        engage_probability = self.adherence - self.fatigue

        if random.random() > engage_probability:
            return "ignore"

        # Nudge effectiveness depends on motivation and addiction
        nudge_quality = {
            "delay_challenge": 0.55,
            "breathing_exercise": 0.45,
            "walk": 0.50,
            "drink_water": 0.35,
            "ai_coach_message": 0.50,
            "social_support": 0.55,
            "educational_reminder": 0.30
        }

        success_probability = (
            nudge_quality[nudge] * self.motivation * (1 - self.addiction * 0.5)
        )

        # High risk makes success harder
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
        Reinforcement learning update:
        Nudges that get better outcomes become more likely later.
        """

        reward = REWARD[response]

        self.nudge_counts[nudge] += 1
        count = self.nudge_counts[nudge]

        old_value = self.nudge_values[nudge]
        new_value = old_value + (reward - old_value) / count

        self.nudge_values[nudge] = new_value

    def update_feedback_loops(self, response):
        """
        Positive and negative feedback loops.
        """

        if response in ["skip", "delay"]:
            # Positive feedback loop:
            # success increases self-efficacy/motivation
            self.motivation = min(1.0, self.motivation + 0.015)
            self.fatigue = max(0.0, self.fatigue - 0.01)

        elif response == "ignore":
            # Negative feedback loop:
            # too many ignored reminders increase fatigue
            self.fatigue = min(1.0, self.fatigue + 0.04)

        elif response == "use":
            # Avoid shame, but motivation may slightly decrease
            self.motivation = max(0.0, self.motivation - 0.005)

    def check_dropout(self):
        """
        Dropout happens when fatigue is high and motivation is low.
        """

        dropout_probability = 0.005 + 0.08 * self.fatigue + 0.04 * (1 - self.motivation)

        if random.random() < dropout_probability:
            self.active = False


# -----------------------------
# 3. Simulation
# -----------------------------

def simulate(n_users=200, days=30, algorithm=True):
    users = []

    user_type_names = list(USER_TYPES.keys())

    for _ in range(n_users):
        user_type = random.choice(user_type_names)
        users.append(User(user_type))

    results = []

    contexts = ["stress", "boredom", "after_meal", "social", "study"]

    for day in range(1, days + 1):
        for user_id, user in enumerate(users):

            if not user.active:
                results.append({
                    "day": day,
                    "user_id": user_id,
                    "user_type": user.user_type,
                    "active": False,
                    "snus_used": np.nan,
                    "nudges_sent": 0,
                    "skips": 0,
                    "delays": 0,
                    "ignores": 0
                })
                continue

            daily_cravings = int(
                user.baseline_use *
                (0.8 + user.stress * 0.5 + user.addiction * 0.4)
            )

            snus_used = 0
            nudges_sent = 0
            skips = 0
            delays = 0
            ignores = 0

            for _ in range(daily_cravings):
                context = random.choice(contexts)
                risk = user.predict_risk()

                # Tracking-only baseline
                if not algorithm:
                    use_probability = 0.45 + 0.40 * user.addiction + 0.20 * user.stress - 0.25 * user.motivation
                    use_probability = max(0.05, min(0.95, use_probability))

                    if random.random() < use_probability:
                        snus_used += 1

                    continue

                # Algorithmic support condition
                if risk < 0.35:
                    # Low risk: no interruption
                    use_probability = 0.35 + 0.35 * user.addiction
                    if random.random() < use_probability:
                        snus_used += 1

                elif risk < 0.65:
                    # Medium risk: light nudge
                    nudges_sent += 1
                    nudge = user.choose_nudge(context)
                    response = user.respond_to_nudge(risk, nudge)
                    user.update_learning(nudge, response)
                    user.update_feedback_loops(response)

                    if response == "skip":
                        skips += 1
                    elif response == "delay":
                        delays += 1
                    elif response == "ignore":
                        ignores += 1
                        snus_used += 1
                    else:
                        snus_used += 1

                else:
                    # High risk: JITAI intervention
                    nudges_sent += 1
                    nudge = user.choose_nudge(context)
                    response = user.respond_to_nudge(risk, nudge)
                    user.update_learning(nudge, response)
                    user.update_feedback_loops(response)

                    if response == "skip":
                        skips += 1
                    elif response == "delay":
                        delays += 1
                    elif response == "ignore":
                        ignores += 1
                        snus_used += 1
                    else:
                        snus_used += 1

            user.check_dropout()

            results.append({
                "day": day,
                "user_id": user_id,
                "user_type": user.user_type,
                "active": user.active,
                "snus_used": snus_used,
                "nudges_sent": nudges_sent,
                "skips": skips,
                "delays": delays,
                "ignores": ignores
            })

    return pd.DataFrame(results)


# -----------------------------
# 4. Run simulation
# -----------------------------

baseline = simulate(n_users=300, days=30, algorithm=False)
adaptive = simulate(n_users=300, days=30, algorithm=True)


# -----------------------------
# 5. Quantitative evaluation
# -----------------------------

def summarize(df, name):
    active_df = df[df["active"] == True]

    avg_day_1 = active_df[active_df["day"] == 1]["snus_used"].mean()
    avg_day_30 = active_df[active_df["day"] == 30]["snus_used"].mean()

    reduction = ((avg_day_1 - avg_day_30) / avg_day_1) * 100

    dropout_rate = 1 - (df[df["day"] == 30]["active"].mean())

    print(f"\n{name}")
    print("-" * 40)
    print(f"Average snus use day 1:  {avg_day_1:.2f}")
    print(f"Average snus use day 30: {avg_day_30:.2f}")
    print(f"Reduction:               {reduction:.1f}%")
    print(f"Dropout rate:            {dropout_rate:.1%}")

    if "nudges_sent" in df.columns:
        print(f"Average nudges per user per day: {active_df['nudges_sent'].mean():.2f}")
        print(f"Average skips per user per day:  {active_df['skips'].mean():.2f}")
        print(f"Average delays per user per day: {active_df['delays'].mean():.2f}")
        print(f"Average ignores per user per day:{active_df['ignores'].mean():.2f}")


summarize(baseline, "Tracking-only baseline")
summarize(adaptive, "Adaptive JITAI + recommender algorithm")


# -----------------------------
# 6. Plot results
# -----------------------------

baseline_daily = baseline.groupby("day")["snus_used"].mean()
adaptive_daily = adaptive.groupby("day")["snus_used"].mean()

plt.figure()
plt.plot(baseline_daily.index, baseline_daily.values, label="Tracking only")
plt.plot(adaptive_daily.index, adaptive_daily.values, label="Adaptive nudging")
plt.xlabel("Day")
plt.ylabel("Average snus portions per user")
plt.title("Simulation of snus reduction over time")
plt.legend()
plt.show()


# By user type
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
