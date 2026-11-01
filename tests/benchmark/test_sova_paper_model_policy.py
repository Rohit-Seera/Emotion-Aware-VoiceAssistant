from benchmark.sova_model_adapters import load_registry

def test_live_registry_contains_only_paper_open_models_plus_weings():
    registry = load_registry()
    ids = [m["id"] for m in registry["models"]]
    assert ids == ["mini_omni", "mini_omni2", "moshi", "llama_omni", "freeze_omni", "glm4_voice", "weings_ai"]
    assert "cascade" not in ids
    assert "free_cascade_local" not in ids

def test_paid_cascade_is_excluded_but_paper_reference_is_preserved():
    registry = load_registry()
    excluded = {m["id"]: m for m in registry["excluded_models"]}
    assert "cascade" in excluded
    assert excluded["cascade"]["status"] == "excluded_from_live_runs_paid_component"
    assert excluded["cascade"]["published_table4_reference_retained"] is True

def test_gpt_oss_is_not_mislabelled_as_a_sova_paper_baseline():
    registry = load_registry()
    assert all(m["id"] != "gpt_oss_120b" for m in registry["models"])
