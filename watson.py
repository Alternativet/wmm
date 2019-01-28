import numpy as np
import datetime
import time

from scipy.special import gamma
# References:
# [1] S. Sra, D. Karp, The multivariate Watson distribution:
#   Maximum-likelihood estimation and other aspects,
#   Journal of Multivariate Analysis 114 (2013) 256-269

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

def pdf(X,mu,kappa):
    (N,p) = X.shape
    cp = gamma(p/2) / (2*np.pi**(p/2) * (kummer(1/2,p/2,kappa)))
    return cp * np.exp(kappa * np.einsum('i,ji->j',mu,X)**2)

def wmm_pdf(X,Mu,Kappa,Pi):
    return np.sum([pi * pdf(X,mu,kappa) for mu,kappa,pi in zip(Mu,Kappa,Pi)],axis=0)

def print_t(s,verbose):
    if verbose:
        print('{:%H:%M:%S}: {}'.format(datetime.datetime.now(),s))

def wmm_fit(X,K,maxkappa=700,maxiter=1000,tol=1e-12,verbose=False):
    (N,p) = X.shape # Get dimensions

    kappa = np.zeros(K);
    mu = np.zeros((K,p));
    bounds = np.zeros((maxiter,K,3));
    llh = np.zeros((maxiter));

    iter = 0
    converged = False
    num = np.zeros((N,K))

    # Initialize posterior
    post = np.random.rand(N,K)
    post = post/np.sum(post,axis=1)[:,np.newaxis]

    while not converged:
        # Maximization
        prior = np.sum(post,axis=0)/N
        for j in range(K): # For each component
            # Compute Sj (4.5)
            Sj = np.einsum('i,ij,ik->jk',post[:,j],X,X) / sum(post[:,j])
            # Compute mu using (4.4) [1], ignoring negative kappa case
            [w,v] = np.linalg.eig(Sj)
            idx = np.argmax(w)
            mu[j,:] = np.real(v[:,idx])
            # Compute Kappa using (4.5) [1]
            r = np.einsum('i,ji,j->',mu[j,:],Sj,mu[j,:]) # mu'*Sj*mu
            r = 0.99 if r > 0.999 else r
            kappa[j] = lower_bound(1/2,p/2,r)
            kappa[j] = maxkappa if kappa[j] > maxkappa else kappa[j]
            bounds[iter,j,:] = [lower_bound(1/2,p/2,r),bound(1/2,p/2,r),upper_bound(1/2,p/2,r)]

        # Expectation
        for j in range(K): # For each component
            cp = gamma(p/2) / (2*np.pi**(p/2) * (kummer(1/2,p/2,kappa[j])))
            # Uses Watson distribution (2.1) [1], compute using 4.3 [1]
            num[:,j] = prior[j] * cp * np.exp(kappa[j] * np.einsum('i,ji->j',mu[j,:],X)**2)

        post = num/np.sum(num,axis=1)[:,np.newaxis]
        llh[iter] = np.sum(np.log(np.sum(num,axis=0)))

        # Convergece
        llh_delta = abs(llh[iter] - llh[iter-1])
        if iter > 0 and abs(llh[iter] - llh[iter-1]) < tol:
            converged = True
        elif iter >= maxiter-1:
            converged = True
            print('Did not converge in maximum iterations')
        iter += 1

        if iter % 10 == 0:
            print_t('Iteration: {:d}, delta llh: {:.2g}'.format(iter,llh_delta),verbose)

    llh = llh[:iter]
    bounds = bounds[:iter]
    return mu,kappa,prior,llh,bounds
