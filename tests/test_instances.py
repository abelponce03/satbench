import lzma

import pandas as pd

from satbench.instances.download import gbd_hash, verify, xz_is_intact
from satbench.instances.select import sample, split_train_test


def pool(n=60):
    return pd.DataFrame(
        dict(
            hash=[f"{i:032x}" for i in range(n)],
            family=["a"] * (n // 2) + ["b"] * (n // 2),
            last_year=[2024 if i % 5 == 0 else 2010 for i in range(n)],
        )
    )


def test_sampling_is_order_independent_and_stable():
    df = pool()
    s1 = sample(df, 10, seed=1, cutoff_year=2023)
    s2 = sample(df.sample(frac=1, random_state=3), 10, seed=1, cutoff_year=2023)
    assert list(s1.hash) == list(s2.hash)
    assert sample(df, 10, seed=2, cutoff_year=2023).hash.tolist() != s1.hash.tolist()


def test_recent_quota_and_stratified_split():
    s = sample(pool(), 10, seed=1, cutoff_year=2023, min_recent_share=0.3)
    assert (s.groupby("family").era.apply(lambda e: (e == "recent").sum()) >= 3).all()
    split = split_train_test(s, 0.5, seed=7)
    counts = s.assign(split=split).groupby(["family", "era", "split"]).size().unstack()
    assert ((counts["test"] - counts["train"]).abs() <= 1).all()  # odd strata cannot split evenly


def test_corrupt_and_mismatched_files_are_rejected(tmp_path):
    good = tmp_path / "g.cnf.xz"
    with lzma.open(good, "wt") as f:
        f.write("p cnf 2 1\n1 2 0\n")
    h = gbd_hash(good)
    assert verify(good, h) == ("ok", "")
    assert verify(good, "0" * 32)[0] == "hash_mismatch"
    bad = tmp_path / "b.cnf.xz"
    bad.write_bytes(good.read_bytes()[:-8])  # truncated download
    assert not xz_is_intact(bad)
    assert verify(bad, h)[0] == "corrupt"
