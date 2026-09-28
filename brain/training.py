import numpy as np
from brian2 import Hz, ms, mV, nA

WINDOW = 50 * ms


class Trainer:
    def __init__(self, pipe, eta=0.02, w_max_scale=2.0, w_min=0.0,
                 channel_reward_only=False, reward_norm="word",
                 reward_mode="class"):
        self.pipe = pipe
        # Use KC->MBON plastic synapses if available, else fall back to all synapses
        self.syn = getattr(pipe, 'kc_mbon_syn', None) or pipe.mcns_syn
        self.eta = eta
        self.w0 = np.asarray(self.syn.w / nA)
        self.w_min = np.full_like(self.w0, w_min)
        self.w_max = self.w0 * w_max_scale
        self.channel_reward_only = channel_reward_only
        self.reward_norm = reward_norm
        self.reward_mode = reward_mode
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

    def present_split(self, word):
        organs = {"A": self.pipe.organ_a, "B": self.pipe.organ_b}
        n = len(word)
        k = max(1, n // 2)
        t0 = self.pipe.net.t
        elig_early = np.zeros(len(self.syn.elig))
        for i, symbol in enumerate(word):
            org = organs[symbol]
            org.source.rates = org.rate
            self.pipe.net.run(WINDOW)
            org.source.rates = 0 * Hz
            if i + 1 == k:
                elig_early = np.asarray(self.syn.elig).copy()
        elig_late = np.asarray(self.syn.elig) - elig_early
        return t0, self.pipe.net.t, elig_early, elig_late

    def decision(self, word):
        self.pipe.reset_state()
        self.reset_traces()
        t0, t1 = self.pipe.present_word(word)
        return self.pipe.read_output(t0, t1)

    def reset_traces(self):
        self.syn.elig = 0
        for var in ("x_fast", "y_fast", "x_slow", "y_slow", "x", "y"):
            if var in self.syn.variables:
                setattr(self.syn, var, 0)

    def evaluate(self, samples):
        results = []
        for word, label in samples:
            result = self.decision(word)
            results.append((word, label, result))
        return results

    def fit(self, samples, epochs, verbose=False):
        history = []
        for epoch in range(epochs):
            self.reset_traces()
            acc_elig = np.zeros(len(self.syn.elig))
            epoch_ok = 0
            for word, label in samples:
                if self.reward_mode == "profile":
                    self.pipe.reset_state()
                    self.reset_traces()
                    t0, t1, elig_early, elig_late = self.present_split(word)
                    result = self.pipe.read_output(t0, t1)
                    correct = (result == "ACCEPT") == bool(label)
                    epoch_ok += int(correct)
                    goal = 1.0 if label else -1.0
                    # Temporal-profile reward: ACCEPT/REJECT are rewarded for
                    # firing in the mirrored late/early half of the word.
                    rv_e = np.zeros(len(self.syn.elig))
                    rv_e[self.post_acc] = -goal
                    rv_e[self.post_rej] = goal
                    rv_l = np.zeros(len(self.syn.elig))
                    rv_l[self.post_acc] = goal
                    rv_l[self.post_rej] = -goal
                    acc_elig += rv_l * elig_late + rv_e * elig_early
                else:
                    self.pipe.reset_state()
                    self.reset_traces()
                    t0, t1 = self.pipe.present_word(word)
                    result = self.pipe.read_output(t0, t1)
                    correct = (result == "ACCEPT") == bool(label)
                    epoch_ok += int(correct)
                    goal = 1.0 if label else -1.0
                    # Per-channel reward: ACCEPT/REJECT subdivision. With
                    # channel_reward_only, non-readout synapses are untouched.
                    rv = np.zeros(len(self.syn.elig))
                    rv[self.post_acc] = goal
                    rv[self.post_rej] = -goal
                    if not self.channel_reward_only:
                        rv[self.post_ignore] = goal
                    elig_w = np.asarray(self.syn.elig)
                    if self.reward_norm == "word":
                        elig_w = elig_w / len(word)
                    acc_elig += rv * elig_w
            w = np.asarray(self.syn.w / nA) + self.eta * acc_elig
            np.clip(w, self.w_min, self.w_max, out=w)
            self.syn.w = w * nA
            accuracy = epoch_ok / len(samples)
            history.append(accuracy)
            self.reset_traces()
            if verbose:
                print(f"epoch {epoch + 1}/{epochs}: train_acc={accuracy:.3f}")
        return history

    def reset_weights_to_initial(self):
        self.syn.w = self.w0 * nA