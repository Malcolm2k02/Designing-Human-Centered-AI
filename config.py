# Configuration parameters for the Q-learning simulation of snus cessation interventions. 

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