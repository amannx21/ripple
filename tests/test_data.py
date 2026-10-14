import pandas as pd
import pytest

from freshroute.adapters import organizer, synthetic
from freshroute.adapters.tables import AdapterError


def test_synthetic_is_seeded_and_labeled():
    a = synthetic.generate(42)
    b = synthetic.generate(42)
    for name in a:
        pd.testing.assert_frame_equal(a[name], b[name])
        assert (a[name]["data_label"] == "SYNTHETIC").all()
    assert len(a["batches"]) == 25
    assert set(a["batches"]["commodity"]) == {"tomato", "mango", "banana", "guava", "onion", "cabbage"}


def test_base_instance_is_consistent(base_instance):
    assert base_instance.source_label == "SYNTHETIC"
    assert base_instance.validate_references() == []
    assert set(base_instance.temp_forecast) == set(base_instance.lanes)


def _write_minimal(folder, drop_col=None, rename=None):
    t = synthetic.generate(1)
    for name in ["batches", "destinations", "lanes", "prices", "demand"]:
        df = t[name]
        if name == "batches" and drop_col:
            df = df.drop(columns=[drop_col])
        if name == "batches" and rename:
            df = df.rename(columns=rename)
        df.to_csv(folder / f"{name}.csv", index=False)


def test_organizer_missing_field_error_is_clear(tmp_path):
    _write_minimal(tmp_path, drop_col="remaining_life_days")
    with pytest.raises(AdapterError) as exc:
        organizer.load(tmp_path)
    msg = str(exc.value)
    assert "remaining_life_days" in msg and "mapping.yaml" in msg


def test_organizer_mapping_renames_columns(tmp_path):
    _write_minimal(tmp_path, rename={"quantity_kg": "Qty", "remaining_life_days": "ShelfLife"})
    (tmp_path / "mapping.yaml").write_text(
        "columns:\n  batches:\n    quantity_kg: Qty\n    remaining_life_days: ShelfLife\n"
    )
    inst = organizer.load(tmp_path)
    assert inst.source_label == "ORGANIZER"
    assert len(inst.batches) == 25
    assert any("hubs derived" in n for n in inst.notes)


def test_organizer_missing_table_error(tmp_path):
    _write_minimal(tmp_path)
    (tmp_path / "lanes.csv").unlink()
    with pytest.raises(AdapterError, match="lanes"):
        organizer.load(tmp_path)
