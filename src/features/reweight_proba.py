class ReweightProba:
    def __init__(self, reweight):
        self.reweight = reweight
        
    def fit(self, pi, q1, r1):
        self.pi = pi
        self.q1 = q1
        self.r1 = r1
        
        if self.reweight == True:
            r0 = 1 - r1
            q0 = 1 - q1
            tot = self.pi * (q1/r1) + (1-self.pi) * (q0/r0)
            w = self.pi*(q1/r1)
            w /= tot
            return w
        else:
            return self.pi

    def transform(self, pi):
        pi_c = pi.copy()
        return self.fit(pi_c, self.q1, self.r1)