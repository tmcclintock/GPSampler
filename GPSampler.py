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
        print "here"
        self.LHC = np.array(lhsmdu.sample(dim, N)).T
        print "now here"

    def evaluate_func(self):
        self.Y = np.array([self.func(np.array(x)) for x in self.LHC])

if __name__ == "__main__":
    #Let's say we have some log-likelihood function
    def llike(x):
        return np.exp(-np.sum((x-0.5)**2)/0.05)

    gps = GPSampler(llike, [[0,1],[0,1]])
    gps.create_LH()
    gps.evaluate_func()

    import matplotlib.pyplot as plt
    #x = np.linspace(0, 1, 50)
    #plt.plot(x, [llike(xi) for xi in x], c='k')
    print gps.LHC.shape
    print gps.Y.shape
    #plt.scatter(gps.LHC[:,0], gps.Y, c='b')
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(gps.LHC[:,0], gps.LHC[:,1], gps.Y)
    plt.show()
