
import matplotlib.pyplot as plt


if __name__ == "__main__":
    from src.read_finance_history import read_files
    df = read_files()
    print(df)

    ax = df["SPY"].plot(title="SPY", label="SPY")
    rm_SPY = df["SPY"].rolling(window=20).mean()
    rm_SPY.plot(label="Rolling Mean", ax=ax)
    plt.show()

