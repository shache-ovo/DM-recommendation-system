"""상호작용 필터링."""

from __future__ import annotations

import pandas as pd


def k_core(interactions: pd.DataFrame, k: int, user_col: str = "user_id", item_col: str = "recipe_id") -> pd.DataFrame:
    """모든 사용자와 아이템이 k개 이상의 상호작용을 가질 때까지 반복해서 걸러낸다.

    한쪽을 지우면 다른 쪽의 개수가 줄어들기 때문에 변화가 없을 때까지 되풀이한다.
    """
    if k <= 1:
        return interactions
    current = interactions
    while True:
        user_counts = current[user_col].map(current[user_col].value_counts())
        item_counts = current[item_col].map(current[item_col].value_counts())
        keep = (user_counts >= k) & (item_counts >= k)
        if keep.all():
            return current
        current = current[keep]
