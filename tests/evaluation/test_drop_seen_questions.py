import json

from experiments.drop_seen_questions import drop_seen_questions, main


ITEMS = [{"query_id": f"nq:{i}", "dataset": "nq", "query": f"q{i}"} for i in range(5)]


def test_removes_only_seen_ids_and_keeps_order():
    kept = drop_seen_questions(ITEMS, {"nq:1", "nq:3", "other:9"})
    assert [it["query_id"] for it in kept] == ["nq:0", "nq:2", "nq:4"]


def test_main_rewrites_items_and_keeps_the_original_even_when_run_twice(tmp_path, monkeypatch):
    import sys

    new, old = tmp_path / "new", tmp_path / "old"
    new.mkdir(); old.mkdir()
    (new / "items.json").write_text(json.dumps(ITEMS))
    (old / "items.json").write_text(json.dumps([ITEMS[0], ITEMS[4]]))
    monkeypatch.setattr(sys, "argv", ["x", "--data", str(new), "--seen-dirs", str(old)])
    main()
    main()  # a second run starts from items_all.json, not from the already filtered file
    assert [it["query_id"] for it in json.loads((new / "items.json").read_text())] == ["nq:1", "nq:2", "nq:3"]
    assert len(json.loads((new / "items_all.json").read_text())) == 5
