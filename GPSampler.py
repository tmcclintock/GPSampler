import numpy as np

class GPSampler(object):
    def __init__(self, func, params, *args):
        self.func   = func
        self.params = params

    def create_LH(self, N=40):
        """Create the latin-hypercube for the parameter space.
        """
        import lhsmdu
        dim = len(self.params)
        self.LHC = np.array(lhsmdu.sample(dim, N)).T

    def evaluate_func(self):
        print self.LHC.shape
        print type(self.LHC[0])
        self.Y = np.array([self.func(np.array(x)) for x in self.LHC])

if __name__ == "__main__":
    #Let's say we have some log-likelihood function
    def llike(x):
        return np.exp(-(x-0.5)**2/0.05)

    gps = GPSampler(llike, [[0,1]])
    gps.create_LH()
    print "here"
    gps.evaluate_func()

    import matplotlib.pyplot as plt
    x = np.linspace(0, 1, 50)
    plt.plot(x, llike(x), c='k')
    plt.scatter(gps.LHC, gps.Y, c='b')
    plt.show()
