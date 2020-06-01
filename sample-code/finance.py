
import pandas as pd
import matplotlib.pyplot as plt

# df_spy = pd.read_csv("spy.csv", usecols=["날짜", "종가"], na_values=["nan"])
# df_spy["날짜"] = df_spy["날짜"].str[0:4] + "-" + df_spy["날짜"].str[6:8] + "-" + df_spy["날짜"].str[10:12]
# df_spy["날짜"] = df_spy["날짜"].astype("datetime64[ns]")
# df_spy = df_spy.set_index("날짜")


if __name__ == "__main__":
    s = "2010-05-01"
    e = "2020-05-29"

    dates = pd.date_range(s, e)
    print(dates)

    # create empty df
    df = pd.DataFrame(index=dates)
    print(df)

    # https://finance.yahoo.com/quote/SPY/history?p=SPY
    files = ["../dataset/SPY.csv", "../dataset/TLT.csv", "../dataset/TIGER200.csv", "../dataset/USD-KRW.csv"]

    for filename in files:
        df_tmp = pd.read_csv(filename, usecols=["Date", "Adj Close"], na_values=["nan"])
        df_tmp = df_tmp.set_index("Date")
        df_tmp = df_tmp.rename(columns={"Adj Close": filename.replace("../dataset/", "").replace(".csv", "")})
        df = df.join(df_tmp)
        df = df.dropna()
    print(df.head())
    # print(df.iloc[10:12])
    # print(df.loc["2020-01-01":"2020-02-01", ["SPY", "TLT"]])
    # print(df.loc["2020-01-01":"2020-02-01"])

    df = df / df.iloc[0]
    print(df.head())
    df.plot()
    plt.show()


