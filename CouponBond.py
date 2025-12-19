from math import exp

class CouponBond:
    def __init__(self, principal, rate, maturity, interest_rate):
        self.principal = principal          # face value of the bond
        self.rate = rate / 100              # coupon rate as a decimal
        self.maturity = maturity            # years to maturity
        self.interest_rate = interest_rate / 100  # market interest rate as a decimal

    def present_value(self, x, n):
        return x / (1 + self.interest_rate) ** n
    
    def present_value_continuous(self, x, t):
        return x * exp(-self.interest_rate*t)
    
    def calculate_price_discrete(self):
        coupon_payment = self.principal * self.rate
        price = 0
        
        # Present value of coupon payments
        for t in range(1, self.maturity + 1):
            price += self.present_value(coupon_payment, t)
        
        # Present value of principal repayment
        price += self.present_value(self.principal, self.maturity)
        
        return price

    def calculate_price_continuous(self):
        coupon_payment = self.principal * self.rate
        price = 0
        
        # Present value of coupon payments
        for t in range(1, self.maturity + 1):
            price += self.present_value_continuous(coupon_payment, t)

        # Present value of principal repayment
        price += self.present_value_continuous(self.principal, self.maturity)

        
        return price
    

if __name__ == '__main__':
    bond = CouponBond(principal=1000, rate=10, maturity=3, interest_rate=4)
    print("Coupon Bond Price Discrete: %.2f" % bond.calculate_price_discrete())
    print("Coupon Bond Price Continuous: %.2f" % bond.calculate_price_continuous())