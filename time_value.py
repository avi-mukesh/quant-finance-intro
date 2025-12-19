from math import exp

def future_discrete_value(x, r, n):
    return x * (1 + r) ** n

def present_discrete_value(x, r, n):
    return x * (1 + r) ** -n

def future_continuous_value(x, r, t):
    return x * exp(r * t)

def present_continuous_value(x, r, t):
    return x * exp(-r * t)

if __name__ == '__main__':
    x = 100 # value of investment
    r = 0.05 # interest rate
    n = 10 # number of periods
    t = 10 # time in years


    print("Future Discrete Value:", future_discrete_value(x, r, n))
    print("Present Discrete Value:", present_discrete_value(x, r, n))
    print("Future Continuous Value:", future_continuous_value(x, r, t))
    print("Present Continuous Value:", present_continuous_value(x, r, t))