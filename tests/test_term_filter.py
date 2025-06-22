import os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
import pytest
from term_filter import is_edit_distance_leq_one


def test_exact_match():
    assert is_edit_distance_leq_one('合同', '合同')


def test_single_replacement():
    assert is_edit_distance_leq_one('劳动', '劳力')


def test_insertion_and_deletion():
    assert is_edit_distance_leq_one('仲裁', '仲裁员')
    assert is_edit_distance_leq_one('审判员', '审判')


def test_distance_gt_one():
    assert not is_edit_distance_leq_one('法院', '检察院')
