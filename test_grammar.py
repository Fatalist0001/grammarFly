from grammar.ab import ABGrammar


def test_is_positive():
    g = ABGrammar()
    for n in range(1, 7):
        positive = g.positive(n)
        assert g.is_positive(positive)
        for word in g.all_words(2 * n):
            assert g.is_positive(word) == (word == positive)


def test_plan_examples():
    g = ABGrammar()
    examples = ["A", "B", "AAB", "ABB", "ABAB", "AAABB", "AAABBBB"]
    all_neg = [w for w, lab in (g.train() + g.validation() + g.test()) if not lab]
    for example in examples:
        assert example in all_neg, example


def test_structure():
    g = ABGrammar()
    train = g.train()
    validation = g.validation()
    test = g.test()

    assert set(w for w, lab in train if lab) == {"AB", "AABB", "AAABBB"}
    assert [w for w, lab in test if lab] == [g.positive(n) for n in (4, 5, 6)]

    for w, lab in train + validation + test:
        assert g.is_positive(w) == lab

    assert not ({w for w, _ in train} & {w for w, _ in test})
    assert not ({w for w, _ in train} & {w for w, _ in validation})
    assert not ({w for w, _ in validation} & {w for w, _ in test})

    assert {len(w) for w, _ in train} == {1, 2, 3, 4, 5, 6}
    assert {len(w) for w, _ in test} == {7, 8, 9, 10, 11, 12, 13}


def test_set_sizes():
    g = ABGrammar()
    tot_neg_len = sum(2 ** L for L in (1, 2, 3, 4, 5, 6)) - 3
    valid_neg_len = sum(1 for L in (1, 2, 3, 4, 5, 6)
                        for w in g.all_words(L)
                        if not ABGrammar.is_positive(w) and ABGrammar.has_ba_reversal(w))
    train_neg_len = tot_neg_len - valid_neg_len
    assert len(g.train()) == 3 + train_neg_len
    assert len(g.validation()) == valid_neg_len
    assert len(g.test()) == 3 + sum(2 ** L for L in (7, 8, 9, 10, 11, 12, 13)) - 3


if __name__ == "__main__":
    g = ABGrammar()
    train = g.train()
    validation = g.validation()
    test = g.test()
    test_is_positive()
    test_plan_examples()
    test_structure()
    test_set_sizes()
    print(f"train={len(train)} (pos {sum(1 for _, l in train if l)}), "
          f"validation={len(validation)}, test={len(test)}")
    print("OK: grammar generator verified")