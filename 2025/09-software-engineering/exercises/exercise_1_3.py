
import math, sys     



def parse_prices(prices):
    result = {}
    for k,v in prices.items():
        try:
            if isinstance(v,str): x=float(v)
            else: x=float(v)
        except Exception as e:
            x=0.0
        result[k]=x
    return result

def stats(nums):
    n=len(nums); 
    if n==0: return {"count":0,"mean":0.0,"stdev":0.0}
    mean=sum(nums)/n
    s=0.0
    for x in nums: s+= (x-mean)**2
    stdev= math.sqrt(s/n) if n>1 else 0.0
    return {"count":n,"mean":mean,"stdev":stdev}

class User:
    def __init__(self,name,scores:list[float]):
        self.name=name; self.scores=scores
    def avg(self):
        if not self.scores: return 0.0
        return sum(self.scores)/len(self.scores)
