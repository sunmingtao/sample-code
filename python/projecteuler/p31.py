p = [1, 2, 5, 10, 20, 50, 100, 200]


def way(money, coin):
    if money == 0 or coin == 1:
        return 1
    max_coin = max(p[i] for i in range(coin))
    return sum(way(money - i * max_coin, coin-1) for i in range(money // max_coin + 1))


print(way(200, 8))
