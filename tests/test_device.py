from clinical_retrieval.device import cuda_available, resolve_device


def test_resolve_device_cpu_explicit():
    assert resolve_device("cpu", log=False) == "cpu"


def test_resolve_device_auto_is_cuda_or_cpu():
    d = resolve_device("auto", log=False)
    assert d in {"cuda", "cpu"}
    if cuda_available():
        assert d == "cuda"
    else:
        assert d == "cpu"


def test_resolve_device_cuda_falls_back_without_gpu(monkeypatch):
    monkeypatch.setattr("clinical_retrieval.device.cuda_available", lambda: False)
    assert resolve_device("cuda", log=False) == "cpu"
