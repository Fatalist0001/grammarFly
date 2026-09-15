class ABGrammar:
    TRAIN_NS = (1, 2, 3)
    TEST_NS = (4, 5, 6)
    TRAIN_NEG_LENGTHS = (1, 2, 3, 4, 5, 6)
    TEST_NEG_LENGTHS = (7, 8, 9, 10, 11, 12, 13)

    @staticmethod
    def positive(n):
        return "A" * n + "B" * n

    @staticmethod
    def is_positive(word):
        n = len(word)
        if n % 2:
            return False
        return word == "A" * (n // 2) + "B" * (n // 2)

    @staticmethod
    def has_ba_reversal(word):
        first_b = word.find("B")
        last_a = word.rfind("A")
        return 0 <= first_b < last_a

    @staticmethod
    def all_words(length):
        for bits in range(1 << length):
            yield "".join("B" if (bits >> i) & 1 else "A" for i in range(length))

    @staticmethod
    def all_negatives(lengths):
        out = []
        for length in lengths:
            out.extend(w for w in ABGrammar.all_words(length) if not ABGrammar.is_positive(w))
        return out

    def train(self):
        positives = [(self.positive(n), True) for n in self.TRAIN_NS]
        return positives + self._negatives(self.TRAIN_NEG_LENGTHS, "no_reversal")

    def validation(self):
        return self._negatives(self.TRAIN_NEG_LENGTHS, "reversal")

    def test(self):
        positives = [(self.positive(n), True) for n in self.TEST_NS]
        return positives + self._negatives(self.TEST_NEG_LENGTHS, "all")

    def _negatives(self, lengths, mode):
        out = []
        for word in self.all_negatives(lengths):
            reversal = self.has_ba_reversal(word)
            if mode == "reversal" and not reversal:
                continue
            if mode == "no_reversal" and reversal:
                continue
            out.append((word, False))
        return out