import numpy as np
from brian2 import Hz, ms, mV, nA

WINDOW = 50 * ms


class Trainer:
    def __init__(self, pipe, eta=0.02, w_max_scale=2.0, w_min=0.0):
        self.pipe = pipe
        # Use KC->MBON plastic synapses if available, else fall back to all synapses
        self.syn = getattr(pipe, 'kc_mbon_syn', None) or pipe.mcns_syn
        self.eta = eta
        self.w0 = np.asarray(self.syn.w / nA)
        self.w_min = np.full_like(self.w0, w_min)
        self.w_max = self.w0 * w_max_scale
        # Per-synapse reward polarity by readout channel of the post neuron
        post_idx = np.asarray(self.syn.j)
        if pipe.readout_w is not None:
            w = np.asarray(pipe.readout_w)
            n = len(w) // 2
            p = w[:n] + w[n:]
            eps = 1e-3 * (np.abs(p).max() + 1e-9)
            self.post_acc = np.isin(post_idx, np.nonzero(p > eps)[0])
            self.post_rej = np.isin(post_idx, np.nonzero(p < -eps)[0])
        else:
            self.post_acc = np.isin(post_idx, np.asarray(pipe.out_acc))
            self.post_rej = np.isin(post_idx, np.asarray(pipe.out_rej))
        self.post_ignore = ~(self.post_acc | self.post_rej)

    def present(self, word):
        organs = {"A": self.pipe.organ_a, "B": self.pipe.organ_b}
        t0 = self.pipe.net.t
        for symbol in word:
            org = organs[symbol]
            org.source.rates = org.rate
            self.pipe.net.run(WINDOW)
            org.source.rates = 0 * Hz
        return t0, self.pipe.net.t

    def decision(self, word):
        self.pipe.reset_state()
        self.syn.elig = 0
        t0, t1 = self.pipe.present_word(word)
        return self.pipe.read_output(t0, t1)

    def evaluate(self, samples):
        results = []
        for word, label in samples:
            result = self.decision(word)
            results.append((word, label, result))
        return results

    def fit(self, samples, epochs, verbose=False):
        history = []
        for epoch in range(epochs):
            self.syn.elig = 0
            acc_elig = np.zeros(len(self.syn.elig))
            epoch_ok = 0
            for word, label in samples:
                self.pipe.reset_state()
                self.syn.elig = 0
                t0, t1 = self.pipe.present_word(word)
                result = self.pipe.read_output(t0, t1)
                correct = (result == "ACCEPT") == bool(label)
                epoch_ok += int(correct)
                goal = 1.0 if label else -1.0
                # Per-channel reward: ACCEPT/REJECT subdivision plus a global
                # R-STDP term for synapses outside the readout channels.
                rv = np.full(len(self.syn.elig), goal)
                rv[self.post_acc] = goal
                rv[self.post_rej] = -goal
                acc_elig += rv * np.asarray(self.syn.elig)
            w = np.asarray(self.syn.w / nA) + self.eta * acc_elig
            np.clip(w, self.w_min, self.w_max, out=w)
            self.syn.w = w * nA
            accuracy = epoch_ok / len(samples)
            history.append(accuracy)
            self.syn.elig = 0
            if verbose:
                print(f"epoch {epoch + 1}/{epochs}: train_acc={accuracy:.3f}")
        return history

    def reset_weights_to_initial(self):
        self.syn.w = self.w0 * nA