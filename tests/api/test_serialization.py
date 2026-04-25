"""Tests for the JSON-safe serialiser used to encode env observations."""
from __future__ import annotations

import numpy as np
import pytest

from verifiable_labs_api.serialization import to_json_safe


def test_passthrough_primitives():
    assert to_json_safe(None) is None
    assert to_json_safe(True) is True
    assert to_json_safe(42) == 42
    assert to_json_safe(3.14) == pytest.approx(3.14)
    assert to_json_safe("hello") == "hello"


def test_complex_split_into_re_im():
    out = to_json_safe(complex(1.5, -2.5))
    assert out == {"re": 1.5, "im": -2.5}


def test_numpy_real_array_to_list():
    arr = np.array([1.0, 2.0, 3.0])
    assert to_json_safe(arr) == [1.0, 2.0, 3.0]


def test_numpy_int_array_to_list():
    arr = np.array([1, 2, 3], dtype=np.int64)
    assert to_json_safe(arr) == [1, 2, 3]


def test_numpy_complex_array_split():
    arr = np.array([1 + 0j, 0 + 1j, -1 + 0j])
    out = to_json_safe(arr)
    assert out["re"] == [1.0, 0.0, -1.0]
    assert out["im"] == [0.0, 1.0, 0.0]


def test_numpy_2d_array_to_nested_list():
    arr = np.zeros((2, 3))
    out = to_json_safe(arr)
    assert out == [[0.0, 0.0, 0.0], [0.0, 0.0, 0.0]]


def test_numpy_scalar_unwrapped():
    assert to_json_safe(np.float64(2.5)) == pytest.approx(2.5)
    assert to_json_safe(np.int64(7)) == 7


def test_dict_recursion_preserves_keys():
    out = to_json_safe({"x": np.array([1.0]), "y": complex(0, 1)})
    assert out == {"x": [1.0], "y": {"re": 0.0, "im": 1.0}}


def test_list_and_tuple_recursion():
    out = to_json_safe([np.float64(1.0), (np.int32(2), "z")])
    assert out == [1.0, [2, "z"]]


def test_set_to_sorted_list():
    assert to_json_safe({3, 1, 2}) == [1, 2, 3]
