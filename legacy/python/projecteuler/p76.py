p = [i for i in range(1, 101)]


def way(money, coin):
    if money == 0 or coin == 1:
        return 1
    max_coin = max(p[i] for i in range(coin))
    return sum(way(money - i * max_coin, coin-1) for i in range(money // max_coin + 1))


print(way(100, 99))