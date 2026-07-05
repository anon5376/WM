from wm.runtime import select_device, smoke_report


def test_torch_import_and_device_fallback():
    report = smoke_report()
    assert report["torch_version"]
    assert report["selected_device"] in {"mps", "cpu"}
    assert isinstance(report["mps_available"], bool)


def test_select_device_cpu_is_explicit():
    info = select_device("cpu")
    assert info.selected == "cpu"
    assert info.reason == "requested cpu"

