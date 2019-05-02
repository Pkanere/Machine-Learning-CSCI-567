from __future__ import print_function
import numpy as np


class HMM:

    def __init__(self, pi, A, B, obs_dict, state_dict):
        """
        - pi: (1*num_state) A numpy array of initial probabilities. pi[i] = P(X_1 = s_i)
        - A: (num_state*num_state) A numpy array of transition probabilities. A[i, j] = P(X_t = s_j|X_t-1 = s_i))
        - B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, o] = P(Z_t = z_o| X_t = s_i)
        - obs_dict: (num_obs_symbol*1) A dictionary mapping each observation symbol to their index in B
        - state_dict: (num_state*1) A dictionary mapping each state to their index in pi and A
        """
        self.pi = pi
        self.A = A
        self.B = B
        self.obs_dict = obs_dict
        self.state_dict = state_dict

    # TODO
    def forward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(X_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(X_t = s_j|X_t-1 = s_i))
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, o] = P(Z_t = z_o| X_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - alpha: (num_state*L) A numpy array delta[i, t] = P(X_t = s_i, Z_1:Z_t | λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        alpha = np.zeros([S, L])
        ###################################################
        # Edit here
        ###################################################
        #print(self.A)
        #print(self.B)
        #print(self.pi)
        for i in range(S):
            alpha[i][0] = self.pi[i]*self.B[i][self.obs_dict[Osequence[0]]]
        #print(alpha)
        
        for j in range(1,L):
            key = Osequence[j]
            index = self.obs_dict[key]
            for i in range(S):
                #print([self.A[k][i] * alpha[k][j-1] for k in range(S)])
                alpha[i][j] = self.B[i][index] * sum([self.A[k][i] * alpha[k][j-1] for k in range(S)])
            #print(alpha)
        
        return alpha

    # TODO:
    def backward(self, Osequence):
        """
        Inputs:
        - self.pi: (1*num_state) A numpy array of initial probailities. pi[i] = P(X_1 = s_i)
        - self.A: (num_state*num_state) A numpy array of transition probailities. A[i, j] = P(X_t = s_j|X_t-1 = s_i))
        - self.B: (num_state*num_obs_symbol) A numpy array of observation probabilities. B[i, o] = P(Z_t = z_o| X_t = s_i)
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - beta: (num_state*L) A numpy array gamma[i, t] = P(Z_t+1:Z_T | X_t = s_i, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        beta = np.zeros([S, L])
        ###################################################
        # Edit here
        ###################################################
        for i in range(S):
            beta[i][L-1] = 1
        #print(beta)
        
        for j in range(L-2,-1,-1):
            key = Osequence[j+1]
            index = self.obs_dict[key]
            for i in range(S):
               
                beta[i][j] = sum([self.A[i][k] * self.B[k][index] * beta[k][j+1] for k in range(S)])
            
        #print(beta)
        return beta

    # TODO:
    def sequence_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: A float number of P(Z_1:Z_T | λ)
        """
        prob = 0
        ###################################################
        # Edit here
        ###################################################
        L=len(Osequence)
        alpha = self.forward(Osequence)
        prob = sum([alpha[k][L-1] for k in range(len(alpha))])
        return prob

    # TODO:
    def posterior_prob(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - prob: (num_state*L) A numpy array of P(X_t = i | O, λ)
        """
        S = len(self.pi)
        L = len(Osequence)
        prob = np.zeros([S, L])
        ###################################################
        # Edit here
        ###################################################
        alpha = self.forward(Osequence)
        beta = self.backward(Osequence)
        seq_prob = self.sequence_prob(Osequence)
        
        for i in range(S):
            for j in range(L):
                prob[i][j] = (alpha[i][j]*beta[i][j])/seq_prob
                
        return prob

    # TODO:
    def viterbi(self, Osequence):
        """
        Inputs:
        - Osequence: (1*L) A numpy array of observation sequence with length L

        Returns:
        - path: A List of the most likely hidden state path k* (return state instead of idx)
        """
        path = []
        ###################################################
        # Edit here
        ###################################################
        S = len(self.pi)
        L = len(Osequence)
        delta = np.zeros([S, L])
        states = np.zeros([S, L])
        
        key_list = list(self.state_dict.keys()) 
        val_list = list(self.state_dict.values()) 
        
        for i in range(S):
            delta[i][0] = self.pi[i]*self.B[i][self.obs_dict[Osequence[0]]]
            states[i][0] = i
        #print(alpha)
        
        for j in range(1,L):
            key = Osequence[j]
            index = self.obs_dict[key]
            for i in range(S):
                #print([self.A[k][i] * alpha[k][j-1] for k in range(S)])
                prev = np.array([self.A[k][i] * delta[k][j-1] for k in range(S)])
                delta[i][j] = self.B[i][index] * np.max(prev)
                states[i][j] = np.argmax(prev)
            #print(alpha)
        #print(delta)
        #print(states)
        
        path = [0 for i in range(L)]
        path[L-1] = np.argmax(np.array([delta[i][L-1] for i in range(S)]))
        for i in range(L-2,-1,-1):
            #print(path[i+1])
            path[i] = states[int(path[i+1])][i+1]
        #print(path)
        
        for i in range(L):
            path[i] = key_list[val_list.index(int(path[i]))]
            
        return path
