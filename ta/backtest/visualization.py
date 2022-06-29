"""Visualization functions for backtesting."""
import matplotlib.pyplot as plt


def plot_all_simulated_cumilative_returns(simulated_stocks):

    simulated_stocks.cumsum().plot(legend=False, linewidth=1,
                                   alpha=0.2, color='green', figsize=(12, 8))
    plt.title('Simulated Returns based on Detrended Stock Returns')
    plt.xlabel('Days')
    plt.ylabel('Cumilative Returns')
    plt.show()


def plot_detrended_vs_normal_cumilative_returns(logreturns, detrended_logreturns):

    logreturns.cumsum().plot(label='Returns', figsize=(12, 8))
    detrended_logreturns.cumsum().plot(label='Detrended Returns', figsize=(12, 8))
    plt.title('Effect of Detrending on Buy and Hold Strategy Cumilative Return')
    plt.xlabel('Date')
    plt.ylabel('Cumilative Return')
    plt.xticks(rotation=45)
    plt.legend()
    plt.show()


def plot_all_strategy_returns(strategy_range, strategy_returns):
    plt.bar(strategy_range, strategy_returns)
    plt.title('All strategies yearly average returns')
    plt.xlabel('Strategy')
    plt.ylabel('Yearly Average Return')
    plt.show()


def plot_simulation_test(all_returns, test_return, bins=50):
    plt.hist(all_returns, bins=50)
    plt.axvline(x=test_return, color='r', linestyle='--')
    plt.title('Simulated Returns vs. Strategy Return')
    plt.xlabel('Yearly Avg Returns')
    plt.ylabel('Frequency')
    plt.show()


def print_p_upper(all_returns, test_return):
    numer_of_simulations_run = len(all_returns)
    number_of_times_strategy_outperform_simuation = (
        all_returns >= test_return).sum()
    p_upper = (number_of_times_strategy_outperform_simuation + 1) / \
        (numer_of_simulations_run + 1)
    return print('-'*10, '\n', 'p value:', p_upper)
