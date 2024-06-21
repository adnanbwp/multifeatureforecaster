import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import math


def run_simulation(past_throughput, feature_stories, simultaneous_features, number_of_simulations, days_to_simulate):
    results = []
    feature_completions = [[] for _ in range(len(feature_stories))]

    for _ in range(number_of_simulations):
        remaining_stories = feature_stories.copy()
        completed_features = 0
        in_progress = [0] * simultaneous_features

        for day in range(days_to_simulate):
            if not remaining_stories and all(f == 0 for f in in_progress):
                break

            daily_throughput = random.choice(past_throughput)
            throughput_per_feature = daily_throughput / \
                min(simultaneous_features, len(remaining_stories) +
                    sum(1 for f in in_progress if f > 0))

            for i in range(simultaneous_features):
                if in_progress[i] > 0:
                    if throughput_per_feature >= in_progress[i]:
                        throughput_per_feature -= in_progress[i]
                        feature_completions[completed_features].append(day + 1)
                        completed_features += 1
                        in_progress[i] = 0
                    else:
                        in_progress[i] -= throughput_per_feature
                        throughput_per_feature = 0

                if in_progress[i] == 0 and remaining_stories:
                    in_progress[i] = remaining_stories.pop(0)
                    if throughput_per_feature >= in_progress[i]:
                        throughput_per_feature -= in_progress[i]
                        feature_completions[completed_features].append(day + 1)
                        completed_features += 1
                        in_progress[i] = 0
                    else:
                        in_progress[i] -= throughput_per_feature
                        throughput_per_feature = 0

                if throughput_per_feature == 0:
                    break

        results.append((completed_features, day + 1))

    return results, feature_completions


def analyze_results(results, feature_completions, feature_stories, days_to_simulate, start_date):
    df_results = pd.DataFrame(
        results, columns=['Completed Features', 'Days Taken'])

    print(
        f"Average completed features: {df_results['Completed Features'].mean():.2f}")

    features_85_percent = sum(1 for completions in feature_completions if len(
        completions) / len(results) >= 0.85)
    print(f"85th percentile of completed features: {features_85_percent}")

    print("\nFeature Completion Forecasts:")

    for i, completions in enumerate(feature_completions):
        if completions:
            completion_day = int(np.percentile(completions, 85))
            completion_prob = len(completions) / len(results) * 100
            forecast_date = start_date + timedelta(days=completion_day)
            print(f"Feature {i+1} (Stories: {feature_stories[i]}):")
            print(f"  Estimated completion day: {completion_day}")
            print(
                f"  Forecasted completion date: {forecast_date.strftime('%Y-%m-%d')}")
            print(f"  Probability of completion: {completion_prob:.2f}%")
        else:
            print(f"Feature {i+1} (Complexity: {feature_stories[i]}):")
            print("  Not completed in any simulation")


def create_histograms(results, number_of_simulations, days_to_simulate):
    completed_features = [r[0] for r in results]

    plt.figure(figsize=(10, 6))
    plt.hist(completed_features, bins=range(min(completed_features), max(completed_features) + 2, 1),
             edgecolor='black')
    plt.xlabel('Number of Completed Features')
    plt.ylabel('Frequency')
    plt.title(
        f'Histogram of Completed Features - ({number_of_simulations} Simulations)')
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(10, 6))
    counts, bins = np.histogram(completed_features, bins=range(
        min(completed_features), max(completed_features) + 2, 1))
    cum_counts = np.cumsum(counts[::-1])[::-1]
    cum_percentage = cum_counts / cum_counts[0] * 100
    plt.bar(bins[:-1], cum_percentage, width=np.diff(bins), edgecolor='black')
    plt.xlabel(f'Number of Completed Features in {days_to_simulate} Days')
    plt.ylabel('Cumulative Percentage')
    plt.title(
        f'Cumulative Probability of Completed Features in {days_to_simulate} Days')
    plt.grid(True)
    plt.ylim(0, 100)

    plt.axhline(y=85, color='red', linestyle='--', linewidth=2)
    plt.text(plt.xlim()[1], 85, '85th percentile',
             horizontalalignment='right', verticalalignment='bottom', color='red')

    plt.show()


def main():
    # Input parameters

    # past daily throughput i.e. number of tasks / stories completed per day
    past_throughput = [1, 5, 0, 2, 3, 0, 1, 1, 0, 2, 1, 0,
                       0, 4, 1, 1, 0, 1, 0, 2, 1, 0, 2, 0, 1, 2, 1, 0, 0, 3]

    # number of stories in each feature. Order of the list also sets feature priority
    feature_stories = [5, 8, 7, 6, 9, 12, 3, 8, 7, 9]

    # number of features the team will work on in parallel
    simultaneous_features = 3

    #  number of simulations to perform for the statistical analysis
    number_of_simulations = 10000

    # number of days to forecast, i.e. if we want to predict the number of completed features in the next 45 days, use 45 here.
    days_to_simulate = 45

    #  the start date of the simulation. This is used to calculate the dates for each day in the simulation
    start_date = datetime(2024, 6, 21)

    # Run simulation
    results, feature_completions = run_simulation(
        past_throughput, feature_stories, simultaneous_features, number_of_simulations, days_to_simulate)

    # Analyze and visualize results
    analyze_results(results, feature_completions,
                    feature_stories, days_to_simulate, start_date)
    create_histograms(results, number_of_simulations, days_to_simulate)


if __name__ == "__main__":
    main()
