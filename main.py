import numpy as np
import matplotlib.pyplot as plt

# ------------------------------------------------------------------------------------------------------------------------------------
class SRM:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.last_spike = np.ones(self.neurons, dtype=float) * -1000000
        self.first_spike = np.ones(self.neurons, dtype=int) * -1

    def eta(self, s):
        return - self.eta_reset*np.exp(-s/self.t_membrane)

    def eps(self, s):
        return (1/(1-self.t_current/self.t_membrane))*(np.exp(-s/self.t_membrane) - np.exp(-s/self.t_current))

    def eps_matrix(self, k, size):
        matrix = np.zeros((self.neurons, size), dtype=float)
        for i in range(k):
            matrix[:, i] = self.eps(k-i)
        return matrix

    def roulette(self, array):
        r = random.uniform(0,1.0)
        c = 0.0
        index=0
        for i in range(len(array)):
            c+=array[i]
            if c > r:
                index=i
                break
        return index

    def agrel_update(self, reward, index):
        pass

    def check_spikes(self, spiketrain, weights, t, last_potential):
        spiketrain_window = spiketrain[:,0:t+1]
        neurons, timesteps = spiketrain_window.shape
        epsilon_matrix = self.eps_matrix(t, timesteps)
        incoming_spikes = np.dot(weights.T, spiketrain_window)
        incoming_potential = np.sum(incoming_spikes * epsilon_matrix, axis=1)
        total_potential = self.eta(np.ones(neurons)*t - self.last_spike) + incoming_potential
        neurons_high_current =  np.where(np.logical_and(total_potential >= self.threshold, last_potential < self.threshold))
        spiketrain[neurons_high_current, t] = True
        spiking_neurons = np.where(spiketrain[:, t])
        self.last_spike[spiking_neurons] = t
        
        for x in spiking_neurons[0]:
            if self.first_spike[x] == -1:
                self.first_spike[x] = t
        return total_potential

# ------------------------------------------------------------------------------------------------------------------------------------
def feedlayout(inputs, hidden, outputs):
    N = inputs+1+hidden+1+outputs
    weights = np.zeros([N,N])
    for i in range(inputs+1):
        weights[i,inputs+1:inputs+hidden+2] = np.random.uniform(-1., 1., (1,hidden+1))
    for i in range(inputs+1,inputs+hidden+2):
        weights[i,inputs+hidden+2:] = np.random.uniform(-1., 1., (1,outputs))
    return weights

def run_epoch(w, N, outputs, T, case):
    model = SRM(neurons=N, threshold=0.4, t_current=0.3, t_membrane=20, eta_reset=5)
    
    last_potential = []
    pot = []
    s = np.zeros([N, T])
    s[0,XORA[case]] = 1
    s[1,XORB[case]] = 1
    
    # bias
    s[2,0] = 1
    s[inputs+hidden+1,0] = 1
    
    for t in range(T):
        last_potential = model.check_spikes(s, w, t, last_potential)
        pot.append(last_potential[-outputs:])

    # Find first-spike times of outputs
    print model.first_spike
    # print s
    # plt.plot(pot)
    # plt.show()

# ------------------------------------------------------------------------------------------------------------------------------------

if __name__ == "__main__":
    inputs = 2
    hidden = 10
    outputs = 2
    N = inputs+1+hidden+1+outputs

    T = 17
    EPOCHS = 1
    
    W = feedlayout(inputs, hidden, outputs)
    
    XORA = [0,0,6,6]
    XORB = [0,6,0,6]

    for epoch in range(EPOCHS):
        for case in range(4):
            run_epoch(W, N, outputs, T, case)