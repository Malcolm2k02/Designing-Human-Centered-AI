from config import Q_TABLE, CONTEXTS, CONTEXT_RISK, OBSERVATION_MODEL, NUDGES, REWARD, ALPHA, GAMMA, USER_TYPES
from utils import discretize
import random
import numpy as np

class User:
    """Class representing a user in the snus cessation intervention simulation, including their characteristics, behavior, and learning updates."""
    def __init__(self, user_type):
        """Initializes a User object with characteristics based on the specified user type,
          including baseline snus use, motivation, addiction level, stress, adherence,
            self-efficacy, and a hidden trigger profile."""
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
        """Returns the trigger with the highest belief probability as the most likely trigger for the user."""
        return max(self.trigger_beliefs, key=self.trigger_beliefs.get)

    def get_state(self, estimated_risk):
        """Returns the current state representation for the user, including user type, inferred trigger,
          risk level, fatigue level, and strategy."""
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
        """Updates the user's beliefs about their triggers based on the observed trigger and their response to a nudge"""
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
        """Calculates the user's actual risk of using snus based on their characteristics and the true triggers they are experiencing."""
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
        """Estimates the user's risk of using snus based on their beliefs about their triggers and their characteristics."""
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
        """Chooses a nudge to send to the user based on the current estimated risk and an epsilon-greedy strategy for exploration."""
        state = self.get_state(estimated_risk)

        if state not in Q_TABLE:
            Q_TABLE[state] = {nudge: 0.0 for nudge in NUDGES}

        if random.random() < epsilon:
            return random.choice(NUDGES)

        return max(Q_TABLE[state], key=Q_TABLE[state].get)

    def respond_to_nudge(self, actual_risk, nudge):
        """Determines the user's response to a nudge based on their actual risk and the type of nudge received,
          as well as their characteristics and current state."""
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
        """Updates the Q-table based on the user's response to a nudge, using the Q-learning algorithm."""
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
        """Updates the user's internal states such as motivation, 
            self-efficacy, craving, and fatigue based on their response 
            to a nudge and the type of nudge received."""
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
        """Determines whether the user drops out of the intervention based on their fatigue,
          motivation, and self-efficacy levels, as well as a base dropout probability."""
        dropout_probability = (
            0.002 +
            0.03 * self.fatigue +
            0.02 * (1 - self.motivation) +
            0.02 * (1 - self.self_efficacy)
        )

        dropout_probability = max(0.0, min(0.12, dropout_probability))

        if random.random() < dropout_probability:
            self.active = False
