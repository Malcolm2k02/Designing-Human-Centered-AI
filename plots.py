import matplotlib.pyplot as plt
import pandas as pd

"""File for generating plots to visualize the results of the snus cessation intervention simulation, including comparisons between the adaptive recommender and baseline conditions, as well as analyses by user type and inferred triggers."""

def make_all_plots(baseline, adaptive, adaptive_events):
    """Generates a series of plots to visualize the results of the snus cessation intervention simulation,
        including comparisons between the adaptive recommender and baseline conditions,
        as well as analyses by user type and inferred triggers."""
    
    plt.rcParams.update({
        "figure.figsize": (9, 5),
        "axes.grid": True,
        "grid.alpha": 0.3,
        "font.size": 11
    })

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

    money_daily_by_type["cumulative_money_saved"] = (
        money_daily_by_type
        .groupby("user_type")["money_saved"]
        .cumsum()
    )

    plt.figure()
    for user_type in money_daily_by_type["user_type"].unique():
        subset = money_daily_by_type[money_daily_by_type["user_type"] == user_type]
        plt.plot(subset["day"], subset["cumulative_money_saved"], linewidth=2, label=user_type)

    plt.xlabel("Simulation day")
    plt.ylabel("Cumulative average money saved per active user (€)")
    plt.title("Cumulative Estimated Money Saved by User Type")
    plt.legend(title="User type", fontsize=9)
    plt.tight_layout()
    plt.show()

    trigger_counts = adaptive_active["inferred_trigger"].value_counts()

    plt.figure()
    plt.bar(trigger_counts.index, trigger_counts.values)
    plt.xticks(rotation=25)
    plt.xlabel("Most likely inferred trigger")
    plt.ylabel("Number of active users")
    plt.title("Distribution of Inferred User Triggers")
    plt.tight_layout()
    plt.show()

    trigger_effect = (
        adaptive_active
        .groupby(["day", "inferred_trigger"])["snus_used"]
        .mean()
        .reset_index()
    )

    plt.figure()
    for trigger in trigger_effect["inferred_trigger"].unique():
        subset = trigger_effect[trigger_effect["inferred_trigger"] == trigger].copy()
        subset["snus_used_smoothed"] = (
            subset["snus_used"]
            .rolling(window=3, min_periods=1)
            .mean()
        )

        plt.plot(subset["day"], subset["snus_used_smoothed"], linewidth=2, label=trigger)

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
    plt.xticks(range(len(confusion.columns)), confusion.columns, rotation=45, ha="right")
    plt.yticks(range(len(confusion.index)), confusion.index)
    plt.xlabel("Inferred trigger at craving moment")
    plt.ylabel("Actual primary trigger")
    plt.title("Event-Level Trigger Inference Accuracy")
    plt.colorbar(label="Proportion")
    plt.tight_layout()
    plt.show()

    fatigue_by_type = (
        adaptive
        .groupby(["day", "user_type"])["fatigue"]
        .mean()
        .reset_index()
    )

    plt.figure(figsize=(10, 6))
    for user_type in fatigue_by_type["user_type"].unique():
        subset = fatigue_by_type[fatigue_by_type["user_type"] == user_type]
        plt.plot(subset["day"], subset["fatigue"], linewidth=2, label=user_type)

    plt.xlabel("Simulation day")
    plt.ylabel("Average fatigue")
    plt.title("Intervention Fatigue by User Type")
    plt.legend()
    plt.tight_layout()
    plt.show()