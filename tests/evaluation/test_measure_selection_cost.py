import numpy as np
import pytest

from experiments.measure_selection_cost import time_stages


class Clock:
    def __init__(self):
        self.t = 0.0

    def __call__(self):
        return self.t


def test_time_stages_reports_median_p90_and_total():
    clock = Clock()
    items = [{"d": 1.0}, {"d": 2.0}, {"d": 3.0}, {"d": 4.0}]

    def slow(item):
        clock.t += item["d"]

    def fixed(item):
        clock.t += 0.5

    r = time_stages({"a": slow, "b": fixed}, items, clock=clock)
    assert r["n_queries"] == 4
    assert r["stages"]["a"]["median"] == pytest.approx(2.5)
    assert r["stages"]["a"]["p90"] == pytest.approx(float(np.percentile([1, 2, 3, 4], 90)))
    assert r["stages"]["b"]["median"] == pytest.approx(0.5)
    assert r["total"]["median"] == pytest.approx(3.0)  # per-query sum of its stages, then the median


def test_stages_run_in_the_given_order_for_every_item():
    seen = []
    items = [{"i": 0}, {"i": 1}]
    time_stages({"x": lambda it: seen.append(("x", it["i"])), "y": lambda it: seen.append(("y", it["i"]))}, items)
    assert seen == [("x", 0), ("y", 0), ("x", 1), ("y", 1)]
