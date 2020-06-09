
import matplotlib.pyplot as plt
import numpy as np

# daily_ret[t] = (price[t] / price[t-1]) - 1
def compute_daily_returns(df):
    daily_returns = df.copy()
    daily_returns[1:] = (df[1:] / df[:-1].values) - 1
    daily_returns.iloc[0] = 0
    return daily_returns


if __name__ == "__main__":
    from src.read_finance_history import read_files
    df = read_files()
    print(df)
    returns = compute_daily_returns(df)

    # histogram plot
    # returns["SPY"].hist(bins=20, label="SPY")
    # returns["TLT"].hist(bins=20, label="TLT")
    # plt.legend(loc="upper right")
    # plt.show()

    # scatter plot
    returns = compute_daily_returns(df)
    returns.plot(kind="scatter", x="SPY", y="TLT")
    # plt.show()

    beta, alpha = np.polyfit(returns["SPY"], returns["TLT"], 1)
    plt.plot(returns["SPY"], beta*returns["SPY"] + alpha, "-", color="r")
    plt.show()

    print(returns.corr(method="pearson"))