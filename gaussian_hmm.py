
# A few issues in this code could be initialization prior, pi, is not used in alpha step
# P(X) does not seems correct
# the scaling factors for alpha and beta need to be re-written
# np.set_printoptions(threshold=np.inf)

import numpy as np
import pandas as pd
from pandas import Series,DataFrame
from scipy.stats import norm

X = np.random.normal(10, 1, 120)
Y = np.random.normal(0, 1, 120)
Z = np.random.normal(3,2,50)
X = np.concatenate(( X, Y, X, Z))


#The parameters only for Gaussian
A=np.random.rand(3, 3); A = A/A.sum(axis=1)[:, np.newaxis]
phi = [[0, 1.2],[10, 2], [3, 2]]




class gaussian_hmm:
    A = []
    X = []
    k_states = []
    N = []
    phi = []

    alpha = []
    beta = []
    alpha_prime = []
    beta_prime = []
    scaling_c = []

    z_X = []
    likelihood = []
    eata = []

    def __init__(self, A, X, phi):
        self.A = A
        self.X = X
        self.k_states = A.shape[0]
        self.N = X.shape[0]
        self.phi = phi
        self.alpha = np.zeros((self.N, self.k_states))
        self.beta = np.zeros((self.N, self.k_states))
        self.beta[self.N-1,]=1

        self.alpha_prime = self.alpha
        self.beta_prime = self.beta
        self.scaling_c = np.zeros(X.shape[0])

    def get_emission( self, n, state):
        X = self.X
        phi = self.phi
        prior = norm.pdf(X[n],phi[state][0], phi[state][1])
        return(prior)

    def get_alpha_N_1(self, n, state):
        alpha = self.alpha
        A = self.A
        sum = 0

        for i in np.arange(self.k_states):
            sum = sum + alpha[n-1][state]*A[i][state]
        return(sum)

    def get_alpha_prime_N_1(self, n, state):
        alpha_prime = self.alpha_prime
        A = self.A
        sum = 0

        for i in np.arange(self.k_states):
            sum = sum + alpha_prime[n-1][state]*A[i][state]
        return(sum)

    def alpha_recursion(self):
        X = self.X
        N = self.N
        alpha = self.alpha
        k_states = self.k_states

        for state in np.arange(k_states):

            alpha[0][state] = self.get_emission(0,state) #requires changing once we know pi

            for i in np.arange(1,N):
                alpha[i][state] = self.get_emission(i, state) * self.get_alpha_N_1(i,state)
                self.alpha = alpha

    def beta_recursion(self):
        beta = self.beta
        X = self.X
        N = self.N
        A = self.A
        k_states = self.k_states

        for state in np.arange(k_states):

            counter = np.arange(0,N-1)
            counter = counter[::-1]
            for n in counter:
                beta[n][state] = sum(beta[n+1][state]*self.get_emission(n+1, state)*A[state][:])
                self.beta[n][state] = beta[n][state]

    def beta_prime_recursion(self):
        beta_prime = self.beta_prime
        X = self.X
        N = self.N
        A = self.A
        k_states = self.k_states
        cn = self.scaling_c

        counter = np.arange(0, N - 1)
        counter = counter[::-1]

        for n in counter:
            for state in np.arange(k_states):
                beta_prime[n][state] = sum(beta_prime[n+1][state]*self.get_emission(n+1, state)*A[state][:]) +2e-20
            self.beta_prime[n][:] = beta_prime[n][:]/sum(beta_prime[n][:])

    def alpha_prime_recursion(self):
        X = self.X
        N = self.N
        alpha_prime = self.alpha_prime
        k_states = self.k_states
        cn = self.scaling_c


        alpha_prime[0][:] = 1  #change once pi is introduced
        cn[0] = 1

        for i in np.arange(1,N):

            for state in np.arange(k_states):

                alpha_prime[i][state] = 2e-20 + self.get_emission(i, state) * self.get_alpha_prime_N_1(i,state)

            try:

                cn[i] = sum(alpha_prime[i][:])
                alpha_prime[i][:] = alpha_prime[i][:]/cn[i]

            except:
                print(i)

        self.alpha_prime = alpha_prime
        self.scaling_c = cn

    def expectation_step(self):

        self.alpha_recursion()
        self.beta_recursion()
        k_states = self.k_states
        N = self.N
        X = self.X

        beta = self.beta
        alpha = self.alpha

        likelihood = (alpha*beta).sum(axis = 1)       # This may be an error
        z_X = (alpha*beta)/likelihood[:, np.newaxis]

        self.z_X = np.asarray(z_X)
        self.likelihood = likelihood

        eata = ['']

        for n in np.arange(1,N):
            M = np.zeros((k_states, k_states))

            for stateN_1 in np.arange(k_states):
                for stateN in np.arange(k_states):
                    M[stateN_1][stateN]=alpha[n-1][stateN_1]*norm.pdf(X[n],phi[stateN][0], phi[stateN][1])*A[stateN_1][stateN]*beta[n][stateN]

            M = M/ M.sum(axis=1)[:, np.newaxis]
            eata.append(M)

        eata.pop(0)
        self.eata = eata

    def expectation_prime_step(self):

        self.alpha_prime_recursion()
        self.beta_prime_recursion()
        k_states = self.k_states
        N = self.N
        X = self.X

        beta = self.beta_prime
        alpha = self.alpha_prime

        likelihood = (alpha * beta).sum(axis=1)  # This may be an error
        z_X = (alpha * beta) / likelihood[:, np.newaxis]

        self.z_X = np.asarray(z_X)
        self.likelihood = likelihood

        eata = ['']

        for n in np.arange(1, N):
            M = np.zeros((k_states, k_states))

            for stateN_1 in np.arange(k_states):
                for stateN in np.arange(k_states):
                    M[stateN_1][stateN] = 2e-20 + alpha[n - 1][stateN_1] * self.get_emission(n,stateN) * \
                                          A[stateN_1][stateN] * beta[n][stateN]

            M = M / M.sum(axis=1)[:, np.newaxis]
            eata.append(M)

        eata.pop(0)
        self.eata = eata

    def recalculate_A(self):
        eata = self.eata
        k_states = self.k_states
        l = len(eata)
        A = np.zeros((k_states,k_states))

        for i in np.arange(l):
            A = A+eata[i]


        A = A / A.sum(axis=1)[:, np.newaxis]
        self.A = A

    def get_emission_parameters(self):
        X = self.X
        N = self.N
        z_X = self.z_X
        K = self.k_states
        phi = self.phi
        #X = X.reshape(N,1)

        for k in np.arange(K):
            phi[k][0] = sum(z_X[:, k] * X) / sum(z_X[:, k])
            # Sigma is working, need to re-write this code
            phi[k][1] = sum((X - phi[k][0])*(X-phi[k][0])*z_X[:,k])/sum(z_X[:,k])

        self.phi = phi

    def maximization_step(self):
        self.recalculate_A()
        self.get_emission_parameters()


    def fit(self):
        for i in np.arange(5):
            self.expectation_prime_step()
            self.maximization_step()



######Testing scrip here

model = gaussian_hmm(A, X, phi)
model.alpha_prime_recursion()
model.beta_prime_recursion()
#model.alpha_recursion()
#model.beta_recursion()

model.expectation_prime_step()
model.maximization_step()

model.expectation_prime_step()
model.maximization_step()

#print(model.A)
np.set_printoptions(threshold=np.inf)
#print(model.z_X)
print(model.phi)


z_X = model.z_X
k = 0





