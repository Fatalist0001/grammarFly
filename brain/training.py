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
                reward = 1.0 if correct else -1.0
                epoch_ok += int(correct)
                acc_elig += reward * np.asarray(self.syn.elig)
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