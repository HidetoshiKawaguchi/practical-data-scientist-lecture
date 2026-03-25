import math, sys

def discounted(price, rate = 0.1):
    if rate<0 or rate>1:print("invalid")
    return price*(1-rate)
