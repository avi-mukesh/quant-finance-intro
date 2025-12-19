from math import exp

class ZeroCouponBond:
    def __init__(self, principal, maturity, interest_rate):
        # principal amount
        self.principal = principal
        # date to maturity
        self.maturity = maturity
        # market interest for discounting
        self.interest_rate = interest_rate / 100
    
    def present_value(self, x, n):
        return x / (1 + self.interest_rate) ** n
    
    def present_value_continuous(self, x, t):
        return x * exp(-self.interest_rate * t)

    def calculate_price(self):
        return self.present_value(self.principal, self.maturity)
    
if __name__ == '__main__':
    bond = ZeroCouponBond(principal=1000, maturity=2, interest_rate=4)
    print("Zero-Coupon Bond Price: %.2f" % bond.calculate_price())