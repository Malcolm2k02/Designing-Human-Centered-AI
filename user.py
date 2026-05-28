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

COST_PER_PORTION = 0.23

ALPHA = 0.1
GAMMA = 0.9

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

        # The system does not know the user's true triggers.
        # It starts with equal beliefs about all possible triggers.
        self.trigger_beliefs = {
            trigger: 1 / len(CONTEXTS) for trigger in CONTEXTS
        }

        self.trigger_profile = np.random.dirichlet(np.ones(len(CONTEXTS))) 
 
    def most_likely_trigger(self):
        """
        The algorithm acts based on its current belief about the user's trigger,
        not the hidden true trigger.
        """
        return max(self.trigger_beliefs, key=self.trigger_beliefs.get)

    def get_state(self, risk):
        """
        Partially observable state:
        The true trigger is hidden, so the Q-table uses the most likely inferred trigger.
        """
        return (
            self.user_type,
            self.most_likely_trigger(),
            discretize(risk),
            discretize(self.fatigue),
            self.strategy
        )

    def observe_trigger(self, true_triggers):
        """
        The system/user only observes one noisy clue about the true trigger.

        Sometimes it observes one of the true triggers.
        Sometimes it observes an unrelated trigger, representing uncertainty.
        """
        if random.random() < 0.75:
            return random.choice(true_triggers)
        else:
            return random.choice(CONTEXTS)

    def update_trigger_beliefs(self, observed_trigger, response):
        """
        Updates inferred trigger profile after each craving event.
        Stronger update when the user uses snus or ignores the system.
        Weaker update when the user successfully skips or delays.
        """

        if response == "use":
            self.trigger_beliefs[observed_trigger] += 0.20

        elif response == "ignore":
            self.trigger_beliefs[observed_trigger] += 0.12

        elif response == "delay":
            self.trigger_beliefs[observed_trigger] += 0.05

        elif response == "skip":
            self.trigger_beliefs[observed_trigger] += 0.02

        # Slightly decay competing trigger beliefs.
        for trigger in self.trigger_beliefs:
            if trigger != observed_trigger:
                self.trigger_beliefs[trigger] *= 0.98

        total = sum(self.trigger_beliefs.values())

        for trigger in self.trigger_beliefs:
            self.trigger_beliefs[trigger] /= total

    def predict_risk(self, true_triggers):
        """
        Risk is based on hidden true triggers.

        The algorithm does not directly observe these triggers, but the user
        behavior is still affected by them.
        """

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

        return max(0.0, min(1.0, risk))

    def choose_nudge(self, risk, epsilon):
        state = self.get_state(risk)

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
                reward = -1.5

        if response == "ignore":
            reward -= self.fatigue * 0.3

        if next_state not in Q_TABLE:
            Q_TABLE[next_state] = {n: 0.0 for n in NUDGES}

        old_q = Q_TABLE[state][nudge]
        best_next_q = max(Q_TABLE[next_state].values())

        new_q = old_q + ALPHA * (reward + GAMMA * best_next_q - old_q)

        Q_TABLE[state][nudge] = new_q

    def update_feedback_loops(self, response, nudge):
        if response == "skip":
            self.motivation = min(1.0, self.motivation + 0.010)
            self.self_efficacy = min(1.0, self.self_efficacy + 0.010)
            self.craving = max(0.0, self.craving - 0.025)
            self.fatigue = max(0.0, self.fatigue - 0.008)

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
        """
        Softer dropout model so high-risk users do not disappear too quickly.
        """

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