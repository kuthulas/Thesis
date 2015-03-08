import numpy as np

class SRM:
    def __init__(self, **kwargs):
        self.__dict__.update(kwargs)
        self.last_spike = np.ones(self.neurons, dtype=float) * -1000000

    def eta(self, s):
        return - self.eta_reset*np.exp(-s/self.t_membrane)

    def eps(self, s):
        return (1/(1-self.t_current/self.t_membrane))*(np.exp(-s/self.t_membrane) - np.exp(-s/self.t_current))

    def eps_matrix(self, k, size):
        matrix = np.zeros((self.neurons, size), dtype=float)
        for i in range(k):
            matrix[:, i] = self.eps(k-i)
        return matrix

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
        # Softmax function
        # Roulette-Wheel selection
        # AGREL update weights + rewards
        return total_potential

if __name__ == "__main__":
    model = SRM(neurons=3, threshold=1, t_current=0.3, t_membrane=20, eta_reset=5)
    s = np.array([[1, 1, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
    # Automate stimuli
    # Rewrite adjacency matrix
    w = np.array([[0, 0, 1.], [0, 0, 1.], [0, 0, 0]])
    neurons, timesteps = s.shape
    last_potential = [0, 0, 0]

    for t in range(timesteps):
        last_potential = model.check_spikes(s, w, t, last_potential)
    print(s)