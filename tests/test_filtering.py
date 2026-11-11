import pandas as pd

from dmrec.filtering import k_core


def frame(pairs):
    return pd.DataFrame(pairs, columns=["user_id", "recipe_id"])


def test_k_core_removes_cascading_rows():
    # u1,u2 는 r1,r2 를 모두 봤다. u3 는 r1 과 r3 를 봤는데 r3 는 u3 만 봤다.
    # r3 가 빠지면 u3 가 1건만 남아 u3 도 빠져야 한다.
    data = frame([("u1", "r1"), ("u1", "r2"), ("u2", "r1"), ("u2", "r2"), ("u3", "r1"), ("u3", "r3")])
    result = k_core(data, 2)
    assert set(result["user_id"]) == {"u1", "u2"}
    assert set(result["recipe_id"]) == {"r1", "r2"}
    assert result["user_id"].value_counts().min() >= 2
    assert result["recipe_id"].value_counts().min() >= 2


def test_k_core_one_keeps_everything():
    data = frame([("u1", "r1"), ("u2", "r2")])
    assert len(k_core(data, 1)) == 2


def test_k_core_can_empty_the_frame():
    data = frame([("u1", "r1"), ("u2", "r2")])
    assert len(k_core(data, 2)) == 0
