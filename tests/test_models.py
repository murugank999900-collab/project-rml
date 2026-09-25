import pytest

torch = pytest.importorskip("torch")

from rml.models import available_models, build_model, register_model  # noqa: E402


def test_cldnn_v1_registered():
    assert "cldnn_v1" in available_models()


def test_cldnn_v1_forward_shape():
    model = build_model("cldnn_v1", in_channels=4, num_classes=11).eval()
    out = model(torch.randn(3, 4, 128))
    assert out.shape == (3, 11)
    assert torch.isfinite(out).all()


def test_cldnn_v1_default_parameter_count():
    model = build_model("cldnn_v1", in_channels=4, num_classes=11)
    n = sum(p.numel() for p in model.parameters())
    assert 500_000 < n < 900_000


def test_cldnn_v1_accepts_other_inputs_and_params():
    model = build_model("cldnn_v1", in_channels=2, num_classes=5, lstm_hidden=16, pool=1).eval()
    assert model(torch.randn(2, 2, 64)).shape == (2, 5)


def test_cldnn_v1_backward():
    model = build_model("cldnn_v1", in_channels=4, num_classes=11, lstm_hidden=16)
    loss = torch.nn.functional.cross_entropy(model(torch.randn(4, 4, 128)), torch.tensor([0, 1, 2, 3]))
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)


def test_unknown_model_and_duplicate_registration():
    with pytest.raises(KeyError, match="Unknown model"):
        build_model("nope", in_channels=4, num_classes=11)
    with pytest.raises(ValueError, match="already registered"):
        register_model("cldnn_v1")(lambda **kw: None)
