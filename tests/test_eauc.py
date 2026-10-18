import numpy as np
import pandas as pd
import pytest
from scipy.stats import kstest, uniform

from eauc import (
    EccentricityCorrector,
    add_eccentricity,
    dataset_ks,
    eauc,
    eccentricity_error_curve,
    entity_ks,
    load_model_outputs,
    make_probe,
    ml_rus,
    save_model_outputs,
)
from eauc.cli import main


@pytest.fixture
def data():
    rng = np.random.default_rng(0)

    def make(n):
        return pd.DataFrame(
            {
                "user_id": rng.integers(0, 50, n),
                "item_id": rng.integers(0, 30, n),
                "rating": rng.integers(1, 6, n).astype(float),
            }
        )

    train, test = make(4000), make(1000)
    for df in (train, test):
        df["pred"] = np.clip(df["rating"] + rng.normal(0, 0.7, len(df)), 1, 5)
    return train, test


def test_perfect_predictions_have_zero_eauc(data):
    train, test = data
    test = test.assign(pred=test["rating"])
    assert eauc(train, test) == 0.0


def test_eauc_in_unit_range_for_all_methods(data):
    train, test = data
    scores = {m: eauc(train, test, method=m, bins=20) for m in ("ordered", "grouped", "bins")}
    assert all(0 <= s <= 1 for s in scores.values())


def test_eauc_matches_algorithm_1(data):
    train, test = data
    t = add_eccentricity(train, test)
    t = t.sort_values("eccentricity")
    x, y = t["eccentricity"].to_numpy(), (t["rating"] - t["pred"]).abs().to_numpy()
    area = sum(0.5 * (x[j] - x[j - 1]) * (y[j] + y[j - 1]) for j in range(1, len(x)))
    span = t["rating"].max() - t["rating"].min()
    assert eauc(train, test) == pytest.approx(area / span**2)


def test_inputs_are_not_mutated(data):
    train, test = data
    before = test.copy()
    eauc(train, test)
    pd.testing.assert_frame_equal(test, before)


def test_unseen_entities_are_dropped(data):
    train, test = data
    extra = test.iloc[:1].assign(user_id=10_000)
    assert len(add_eccentricity(train, pd.concat([test, extra]))) == len(test)


def test_unknown_method_and_missing_column(data):
    train, test = data
    with pytest.raises(ValueError):
        eauc(train, test, method="nope")
    with pytest.raises(KeyError):
        eauc(train, test.drop(columns="pred"))


def test_custom_columns(data):
    train, test = data
    ren = {"user_id": "drug", "item_id": "cell", "rating": "ic50", "pred": "yhat"}
    expected = eauc(train, test)
    got = eauc(
        train.rename(columns=ren),
        test.rename(columns=ren),
        pred_col="yhat",
        user_col="drug",
        item_col="cell",
        value_col="ic50",
    )
    assert got == pytest.approx(expected)


def test_dataset_ks_matches_scipy(data):
    train, _ = data
    lo, hi = train["rating"].min(), train["rating"].max()
    ref = [kstest(g["rating"], uniform(lo, hi - lo).cdf)[0] for _, g in train.groupby("user_id")]
    ref += [kstest(g["rating"], uniform(lo, hi - lo).cdf)[0] for _, g in train.groupby("item_id")]
    assert dataset_ks(train) == pytest.approx(np.mean(ref))


def test_curve_shape(data):
    train, test = data
    curve = eccentricity_error_curve(train, test, bins=10)
    assert list(curve.columns) == ["eccentricity", "mae", "count"]
    assert curve["count"].sum() == len(add_eccentricity(train, test))


@pytest.mark.parametrize("strategy", ["linear", "random_forest"])
def test_corrector_reduces_dmv_bias(strategy):
    rng = np.random.default_rng(1)
    n = 6000
    df = pd.DataFrame(
        {
            "user_id": rng.integers(0, 80, n),
            "item_id": rng.integers(0, 50, n),
            "rating": rng.integers(1, 6, n).astype(float),
        }
    )
    train, test = df.iloc[:5000].copy(), df.iloc[5000:].copy()
    for d in (train, test):
        dmv = (
            d["user_id"].map(train.groupby("user_id")["rating"].mean())
            + d["item_id"].map(train.groupby("item_id")["rating"].mean())
        ) / 2
        d["pred"] = np.clip(0.8 * dmv.fillna(3) + 0.2 * d["rating"], 1, 5)  # strongly biased to the DMV
    _, probe, test_rest = make_probe(train, test)
    corrector = EccentricityCorrector(strategy).fit(train, probe)
    corrected = corrector.transform(test_rest)
    assert eauc(train, corrected) < eauc(train, test_rest)
    assert corrected["pred"].between(1, 5).all()
    assert len(corrected) == len(test_rest)


def test_corrector_keeps_unseen_predictions(data):
    train, test = data
    _, probe, _ = make_probe(train, test)
    corrector = EccentricityCorrector().fit(train, probe)
    unseen = test.iloc[:3].assign(user_id=99_999, pred=2.5)
    assert (corrector.predict(unseen) == 2.5).all()


def test_cli(tmp_path, data, capsys):
    train, test = data
    train.to_csv(tmp_path / "train.csv", index=False)
    test.to_csv(tmp_path / "test.csv", index=False)
    assert main(["score", "--train", str(tmp_path / "train.csv"), "--test", str(tmp_path / "test.csv")]) == 0
    assert "EAUC=" in capsys.readouterr().out
    assert main(["ks", "--data", str(tmp_path / "train.csv")]) == 0
    assert (
        main(
            [
                "correct",
                "--train",
                str(tmp_path / "train.csv"),
                "--test",
                str(tmp_path / "test.csv"),
                "--out",
                str(tmp_path / "o.csv"),
            ]
        )
        == 0
    )
    out = capsys.readouterr().out
    assert "before" in out and "after" in out and (tmp_path / "o.csv").exists()


def test_entity_ks_is_per_entity(data):
    train, _ = data
    per_user = entity_ks(train, "user_id")
    assert per_user.index.name == "user_id"
    assert len(per_user) == train["user_id"].nunique()
    assert per_user.between(0, 1).all()


def test_ml_rus_equalises_value_counts_inside_groups():
    probe = pd.DataFrame({"rating": [1.0] * 6 + [2.0] * 2 + [1.0] * 3 + [2.0] * 3, "group": [0] * 8 + [1] * 6})
    balanced = ml_rus(probe, probe["group"], "rating")
    assert balanced.groupby(["group", "rating"]).size().to_dict() == {
        (0, 1.0): 2,
        (0, 2.0): 2,
        (1, 1.0): 3,
        (1, 2.0): 3,
    }


def test_save_and_load_model_outputs_number_runs(tmp_path, data):
    train, test = data
    assert save_model_outputs(train, test, "M", "d", {"lr": 0.1}, root=tmp_path) == 1
    assert save_model_outputs(train, test, "M", "d", root=tmp_path) == 2
    loaded_train, loaded_test = load_model_outputs("M", "d", 2, root=tmp_path)
    assert len(loaded_train) == len(train) and len(loaded_test) == len(test)
    assert len((tmp_path / "d" / "M" / "params.txt").read_text().splitlines()) == 2
