import torch

from historian_llm.hrm import (
    FEATURE_DIM,
    answer_features,
    features_tensor,
    hrm_score,
    load_hrm,
    save_hrm,
    train_hrm,
)


def test_answer_features_dim_and_bounds():
    feats = answer_features("gouvernement", "gouvernement")
    assert len(feats) == FEATURE_DIM
    # Perfect match: 1-CER, F1, EM, char-jaccard, prefix all 1; CER == 0.
    assert feats[0] == 1.0 and feats[1] == 1.0 and feats[2] == 1.0
    assert feats[3] == 0.0


def test_features_tensor_shape():
    t = features_tensor([("a", "a"), ("b", "c")])
    assert t.shape == (2, FEATURE_DIM)


def test_train_hrm_learns_orthographic_preference():
    prefs = [
        {"gold": "gouvernement", "chosen": "gouvernement", "rejected": "gouverpement"},
        {"gold": "batavia", "chosen": "batavia", "rejected": "bxtwvia"},
        {"gold": "rijksdaalders", "chosen": "rijksdaalders", "rejected": "rijxdazlders"},
        {"gold": "curacaosche", "chosen": "curacaosche", "rejected": "curxcqosche"},
    ]
    model, history = train_hrm(prefs, epochs=150, lr=0.05, seed=0)
    assert history[-1] < history[0]
    for r in prefs:
        assert hrm_score(model, r["chosen"], r["gold"]) > hrm_score(
            model, r["rejected"], r["gold"]
        )


def test_hrm_save_load_roundtrip(tmp_path):
    prefs = [
        {"gold": "batavia", "chosen": "batavia", "rejected": "btvxia"},
        {"gold": "amsterdam", "chosen": "amsterdam", "rejected": "amztrdxm"},
    ]
    model, _ = train_hrm(prefs, epochs=50, lr=0.05, seed=1)
    path = tmp_path / "hrm.pt"
    save_hrm(model, path)
    loaded = load_hrm(path)
    feats = features_tensor([("batavia", "batavia")])
    with torch.no_grad():
        assert torch.allclose(model(feats), loaded(feats), atol=1e-6)
