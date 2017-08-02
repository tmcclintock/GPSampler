import numpy as np

class GPSampler(object):
    def __init__(self, func, params, *args):
        self.func   = func
        self.params = params

    def create_LH(self, N=40):
        """Create the latin-hypercube for the parameter space.
        """
        import lhsmdu
        self.dim = len(self.params)
        self.N   = N
        print "here"
        self.LHC = np.array(lhsmdu.sample(self.dim, self.N)).T
        print "now here"

    def evaluate_func(self):
        self.Y = np.array([self.func(np.array(x)) for x in self.LHC])

    def train_GP(self):
        import george
        lguess = (np.max(self.LHC, 0) - np.min(self.LHC, 0))/self.N
        kernel = 1.0*george.kernels.ExpKernel(metric=lguess, ndim=self.dim) + george.kernels.WhiteKernel(1.0, ndim=self.dim)
        gp = george.GP(kernel)
        gp.optimize(self.LHC, self.Y)
        print gp.kernel
        self.GP = gp

    def predict(self, t):
        return self.GP.predict(self.Y, t)

if __name__ == "__main__":
    #Let's say we have some log-likelihood function
    def llike(x):
        return np.exp(-np.sum((x-0.5)**2)/0.05)

    gps = GPSampler(llike, [[0,1],[0,1]])
    gps.create_LH()
    gps.evaluate_func()
    gps.train_GP()

    import matplotlib.pyplot as plt
    x = np.linspace(-0.5, 1.5, 50)
    #plt.plot(x, [llike(xi) for xi in x], c='k')
    #mu, cov = gps.predict(x)
    #plt.plot(x, mu, c='r')
    #plt.scatter(gps.LHC[:,0], gps.Y, c='b')    
    #err = np.sqrt(np.diag(cov))
    #plt.fill_between(x, mu-err, mu+err, color='r', alpha=0.2)
    print gps.LHC.shape
    print gps.Y.shape
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.scatter(gps.LHC[:,0], gps.LHC[:,1], gps.Y)

    Nd = 50
    X0 = np.linspace(np.min(x), np.max(x), Nd)
    X1 = np.linspace(np.min(x), np.max(x), Nd)
    D0, D1 = np.meshgrid(X0, X1)
    X = np.array([D0.flatten(), D1.flatten()])
    mu = np.ones_like(D0)
    for i in range(len(X0)):
        for j in range(len(X1)):
            Xij = np.atleast_2d([X0[i], X1[j]])
            muij, cov = gps.predict(Xij)
            mu[i,j] = muij
    print mu.shape
    count = 20
    ax.plot_wireframe(D0, D1, mu, rcount=count, ccount=count, color='r', alpha=0.2)

    plt.show()
