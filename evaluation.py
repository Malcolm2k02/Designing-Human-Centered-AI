"""File for evaluation functions to analyze the results of the snus cessation intervention simulation."""
import random
from config import Q_TABLE


def summarize(df, name):
    """Summarizes the results of the snus cessation intervention simulation for a given DataFrame and condition name,
      including average snus use, reduction percentage, dropout rate, money saved, 
      and average nudges/skips/delays/ignores per user per day."""
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


def print_random_q_values(n=5):
    print("\nRandom learned Q-values:\n")

    trained_states = [
        state for state, actions in Q_TABLE.items()
        if any(value != 0 for value in actions.values())
    ]

    if not trained_states:
        print("No trained Q-values found.")
        return

    random_states = random.sample(trained_states, min(n, len(trained_states)))

    for state in random_states:
        actions = Q_TABLE[state]
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


def trigger_accuracy(adaptive_events):
    accuracy = (
        adaptive_events["actual_primary_trigger"] ==
        adaptive_events["inferred_trigger"]
    ).mean()

    print(f"Event-level trigger inference accuracy: {accuracy:.1%}")
    return accuracy
