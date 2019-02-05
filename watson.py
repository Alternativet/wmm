import numpy as np
import datetime
import time

from scipy.special import gamma
# References:
# [1] S. Sra, D. Karp, The multivariate Watson distribution:
#   Maximum-likelihood estimation and other aspects,
#   Journal of Multivariate Analysis 114 (2013) 256-269

def pdf(X,mu,kappa):
    (N,p) = X.shape
    cp = gamma(p/2) / (2*np.pi**(p/2) * (kummer(1/2,p/2,kappa)))
    return cp * np.exp(kappa * np.einsum('i,ji->j',mu,X)**2)

def wmm_pdf(X,Mu,Kappa,Pi):
    return np.sum([pi * pdf(X,mu,kappa) for mu,kappa,pi in zip(Mu,Kappa,Pi)],axis=0)

def wmm_fit(X,k,maxkappa=700,maxiter=200,tol=1e-4,verbose=False,seed=None):
    if seed is not None:
        np.random.seed(seed)

    # Get dimensions
    (N,p) = X.shape

    # Initialization
    if p == 3:
        mu,kappa,pi = init_parameters(k)
    else:
        mu,kappa,pi = np.zeros((k,p)),np.zeros(k),np.zeros(k)
        beta = np.random.rand(N,k)
        beta = beta/np.sum(beta,axis=1)[:,np.newaxis]
        mu,kappa,pi,_ = m_step(X,mu,kappa,pi,beta,k,N,p,maxkappa)

    # Allocation
    bounds = np.zeros((maxiter,k,3));
    llh = np.zeros((maxiter));
    num = np.zeros((N,k))

    iter = 0
    converged = False

    # EM loop
    while not converged:
        beta,llh[iter] = e_step(X,mu,kappa,pi,num,k,p)
        mu,kappa,pi,bounds[iter,:] = m_step(X,mu,kappa,pi,beta,k,N,p,maxkappa)
        converged = convergence(llh,iter,maxiter,tol,verbose)
        iter += 1

    llh = llh[:iter]
    bounds = bounds[:iter]
    return mu,kappa,pi,llh,bounds

def init_parameters(k):
    # Initialise mu based on "The golden spiral method" https://stackoverflow.com/a/44164075/6843855
    indices = np.arange(0, k, dtype=float) + 0.5
    phi = np.arccos(1 - 2*indices/(2*k))
    theta = np.pi * (1 + 5**0.5) * indices
    mu = np.stack((np.cos(theta) * np.sin(phi), np.sin(theta) * np.sin(phi), np.cos(phi)),axis=1)

    # Initalize kappa
    kappa = np.ones(k)

    # Initialize pi
    pi = np.ones(k)/k

    return mu,kappa,pi

def e_step(X,mu,kappa,pi,num,k,p):
    # Expectation
    for j in range(k): # For each component
        cp = gamma(p/2) / (2*np.pi**(p/2) * (kummer(1/2,p/2,kappa[j])))
        # Uses Watson distribution (2.1) [1], compute using 4.3 [1]
        num[:,j] = pi[j] * cp * np.exp(kappa[j] * np.einsum('i,ji->j',mu[j,:],X)**2)

    beta = num/np.sum(num,axis=1)[:,np.newaxis]
    llh = np.sum(np.log(np.sum(num,axis=1)))

    return beta,llh

def m_step(X,mu,kappa,pi,beta,k,N,p,maxkappa):
    bounds = np.zeros((k,3))

    # Maximization
    pi = np.sum(beta,axis=0)/N
    for j in range(k): # For each component
        # Compute Sj (4.5)
        Sj = np.einsum('i,ij,ik->jk',beta[:,j],X,X,optimize='greedy') / np.sum(beta[:,j])
        # Compute mu using (4.4) [1], ignoring negative kappa case
        [w,v] = np.linalg.eig(Sj)
        idx = np.argmax(w)
        mu[j,:] = np.real(v[:,idx])
        # Compute Kappa using (4.5) [1]
        r = np.einsum('i,ji,j->',mu[j,:],Sj,mu[j,:]) # mu'*Sj*mu
        r = 0.99 if r > 0.999 else r
        kappa[j] = lower_bound(1/2,p/2,r)
        kappa[j] = maxkappa if kappa[j] > maxkappa else kappa[j]
        bounds[j,:] = [lower_bound(1/2,p/2,r),bound(1/2,p/2,r),upper_bound(1/2,p/2,r)]

    return mu,kappa,pi,bounds

def convergence(llh,iter,maxiter,tol,verbose):
    # Convergece
    if iter > 0 and abs((llh[iter] - llh[iter-1])/llh[iter-1]) < tol:
        print_t('Conveged in {} iterations'.format(iter+1),verbose)
        return True
    elif iter >= maxiter-1:
        print_t('Did not converge in maximum iterations')
        return True

    if iter+1 % 10 == 0:
        print_t('Iteration: {:d}, relative llh change: {:.2g}'.format(iter+1,(llh[iter] - llh[iter-1])/llh[iter-1]),verbose)

    return False

# Definition of bounds solutions for Kappa (3.7)(3.8)(3.9) [1]
def lower_bound(a,c,r):
    return (r*c-a)/(r*(1-r))*(1+(1-r)/(c-a))

def bound(a,c,r):
    return (r*c-a)/(2*r*(1-r))*(1+np.sqrt( 1+(4*(c+1)*r*(1-r)) / (a* (c-a)) ) )

def upper_bound(a,c,r):
    return (r*c-a)/(r*(1-r))*(1+r/a)

def kummer(a,c,x):
    # Adapted from https://github.com/yuhuichen1015/SphericalDistributionsRand/blob/master/kummer.m
    tol = 1e-10;
    term = x*a/c;
    f = 1 + term;
    n = 1;
    an = a;
    cn = c;
    nmin = 10;
    while n < nmin or abs(term) > tol:
        n = n + 1;
        an = an + 1;
        cn = cn + 1;
        term = x*term*an/cn/n;
        f = f + term;

    return f

def print_t(s,verbose=True):
    if verbose:
        print('{:%H:%M:%S}: {}'.format(datetime.datetime.now(),s))
