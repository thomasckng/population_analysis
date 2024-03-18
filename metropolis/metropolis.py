import numpy as np
from tqdm import tqdm
from scipy.stats import uniform, multivariate_normal as mn

class Metropolis:
    """
    Metropolis optimisation algorithm. A new point is accepted if and only if p_new > p_old.
    
    Arguments:
        np.ndarray bounds: boundaries for the parameters. Formatted as [[xmin, xmax],[ymin,ymax],...]
        callable pdf:      function to optimise
        int n_burnin:      number of points before drawing a sample
        np.ndarray dx:     std of the multivariate_normal distribution to propose a new point
    """
    def __init__(self, bounds,
                       pdf,
                       burnin = 1e4,
                       dx     = None
                       ):
        self.bounds   = np.atleast_2d(bounds)
        self.pdf      = pdf
        self.n_dim    = len(bounds)
        if dx is not None:
            self.dx   = np.atleast_1d(dx)
        else:
            self.dx   = np.diff(self.bounds, axis = 1).flatten()/100
        self.proposal = mn(np.zeros(self.n_dim), np.identity(self.n_dim)*self.dx**2)
        self.initial  = uniform(self.bounds[:,0], self.bounds[:,1])
        self.burnin   = int(burnin)
    
    def _propose_point(self, x):
        """
        Propose a new point within the boundaries
        
        Arguments:
            np.ndarray x: old point
        
        Return:
            np.ndarray: new point
        """
        y = x + self.proposal.rvs()
        while not (np.prod(self.bounds[:,0] <= y) & np.prod(y <= self.bounds[:,1])):
            y = x + self.proposal.rvs()
        return y
    
    def _sample_point(self):
        """
        Sample a new point
        
        Return:
            np.ndarray: new point
        """
        x = self.initial.rvs()
        for _ in range(self.burnin):
            y = self._propose_point(x)
            if self.pdf(y) > self.pdf(x):
                x = y
        return x
    
    def rvs(self, size = 1):
        """
        Sample points
        
        Arguments:
            int size: number of points to draw
        
        Return:
            np.ndarray: array of samples
        """
        return np.array([self._sample_point() for _ in tqdm(range(int(size)))])
