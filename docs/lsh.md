# LSH: 재료가 비슷한 레시피 검색

MinHash 시그니처에 banding LSH를 적용해, 모든 쌍을 비교하지 않고 재료가 비슷한 레시피를 찾는다.
이 문서는 구현, 조회 방법, 기본 설정에서의 인덱스 규모를 적는다.
(b, r) 설정별 재현율과 속도 비교는 LSH 평가 문서에서 다룬다.

```bash
python3 scripts/build_minhash.py   # 먼저 시그니처를 만든다
python3 scripts/build_lsh.py       # 인덱스를 만들고 data/processed/lsh_neighbors.csv 저장 (약 20초)
```

## 1. 방법

1. 길이 k = b · r인 시그니처를 r개씩 b개의 band로 나눈다.
2. band마다 r개 값을 키로 삼아 레시피를 버킷에 넣는다.
3. 어떤 band에서든 같은 버킷에 든 두 레시피를 **후보 쌍**으로 본다.
4. 후보에 대해서만 시그니처 전체로 Jaccard를 추정해 순위를 매긴다.

Jaccard가 s인 두 레시피가 후보가 될 확률은 1 − (1 − s^r)^b다 [1].
이 확률은 (1/b)^(1/r) 근처에서 가파르게 오른다. r이 크면 문턱이 높아져 후보가 줄고, b가 크면 문턱이 낮아져 후보가 늘어난다.

기본 설정은 b = 32, r = 4(문턱 약 0.42)다. 이 값은 잠정값이고 LSH 평가에서 정한다.

## 2. 구현

`src/dmrec/lsh.py`에 직접 구현했다. 외부 LSH 라이브러리를 쓰지 않는다.

- **band 키**: band의 r개 값(32비트)을 64비트 정수 하나로 접는다. band 번호를 초깃값에 섞어 서로 다른 band의 키가 섞이지 않게 한다. 값이 다른데 키가 같아질 확률은 쌍마다 2⁻⁶⁴ 수준이고, 그 경우에도 후보가 하나 늘 뿐 결과 순위는 시그니처로 다시 계산하므로 틀리지 않는다.
- **버킷**: band마다 키로 정렬한 행 번호 배열과 구간 시작 위치를 저장한다. 파이썬 사전 대신 배열을 써서 메모리를 줄였다.
- **후보 조회**: 레시피가 속한 b개 버킷의 구성원을 모아 중복을 없애고 자기 자신을 뺀다.
- **순위**: 추정 Jaccard 내림차순, 같으면 행 번호 오름차순이다. 실행할 때마다 같은 결과가 나온다.
- **빈 집합**은 인덱스에 넣지 않는다.
- **인덱스에 없는 시그니처**로도 검색할 수 있다(`query_signature`). 새 레시피의 재료 집합으로 비슷한 레시피를 찾는 경우다.

| 함수 | 하는 일 |
|---|---|
| `LSHIndex(bands, rows).build(signatures)` | 인덱스 생성 |
| `index.candidates(row)` | 후보 행 번호 |
| `index.query(row, top_n, min_similarity)` | 후보를 추정 Jaccard로 정렬해 (행 번호, 추정값) 반환 |
| `index.query_signature(signature, ...)` | 인덱스에 없는 시그니처로 검색 |
| `collision_probability(s, bands, rows)` | S-curve 값 |

## 3. recipe_id로 조회하기

`src/dmrec/similar.py`의 `SimilarRecipes`가 행 번호를 `recipe_id`로 바꿔 준다.

```python
import sys
sys.path.insert(0, "src")
from dmrec.config import load_config, resolve_path
from dmrec.similar import SimilarRecipes

config = load_config()
finder = SimilarRecipes.from_processed(resolve_path(config, "processed_dir"), config)

finder.similar(27208, top_n=5)          # [(357008, 0.617), (298927, 0.492), ...]
finder.similar(27208, min_jaccard=0.6)  # 추정 Jaccard 0.6 이상만

# 사용자가 좋아한 레시피들의 유사 레시피를 합친다. 입력 레시피 자신은 빠진다.
liked = [27208, 89204, 39087]
finder.candidates_for(liked, top_n_each=20)   # {후보 recipe_id: 추정 Jaccard 최댓값}
```

- 반환되는 유사도는 정확한 Jaccard가 아니라 MinHash 추정값이다(k = 128에서 RMSE 약 0.04, `docs/minhash.md`).
- 인덱스에 없는 `recipe_id`로 `similar`를 부르면 `KeyError`다. `recipe_id in finder`로 확인할 수 있다.
- 기본 범위는 추천 대상 레시피 37,818개다(`lsh.index_scope: core`). `all`로 바꾸면 전체 231,352개를 넣는다.

미리 계산한 결과는 `data/processed/lsh_neighbors.csv`에 있다. 형식은 `docs/data_schema.md`에 있다.

검색 결과의 예(추정 Jaccard):

| 기준 레시피 | 유사 레시피 |
|---|---|
| to die for crock pot roast | three packs and a roast (0.62), 3 pack pot roast (0.49) |
| creamy cajun chicken pasta | red lobster cajun chicken pasta (0.69), blackened chicken pasta with alfredo sauce (0.45) |

## 4. 기본 설정에서의 인덱스 규모

`reports/tables/lsh_build_summary.json`의 값이다. 추천 대상 레시피 37,818개, b = 32, r = 4.

| 항목 | 값 |
|---|---|
| 후보 쌍 | 10,433,774 (전체 쌍의 1.46%) |
| 레시피당 후보 수 | 중앙값 77, 평균 551.8, 최대 4,526 |
| 후보가 하나도 없는 레시피 | 1,853 (4.9%) |
| 상위 20개 이웃을 다 채운 레시피 | 25,936 (68.6%) |
| 1순위 이웃의 추정 Jaccard(중앙값) | 0.484 |
| 버킷 수 / 최대 버킷 크기 | 925,631 / 2,473 |
| 인덱스 생성 | 수 초 이내 |
| 레시피 하나의 상위 20개 조회 | 약 0.3 ms |

시간은 Claude 작업 공간(CPU 2코어)에서 잰 값이다.

- 전체 쌍의 1.46%만 비교한다. 레시피당 후보 수는 치우쳐 있어서 일부 레시피는 수천 개의 후보를 갖는다. 흔한 재료 하나가 한 band의 해시 함수 4개에서 모두 최솟값이 되면 그 재료를 가진 레시피가 한 버킷에 몰리기 때문이다. 가장 큰 버킷(2,473개)의 레시피는 모두 pepper를 포함한다. 500개 이상인 버킷은 12개다.
- 후보가 없는 레시피가 4.9% 있다. 문턱(약 0.42)보다 비슷한 레시피가 없거나, 있어도 확률적으로 놓친 경우다. 문턱을 낮추면 줄어든다.

## 5. 검증

단위 테스트(`tests/test_lsh.py` 13개, `tests/test_similar.py` 4개):

- 인덱스가 돌려주는 후보가 정의(어떤 band에서든 r개 값이 모두 같은 행)대로 전수 계산한 결과와 정확히 같다. (b, r) 네 가지 조합에서 확인한다.
- band 키가 같은 것과 band 값이 같은 것이 일치한다.
- 같은 시그니처는 항상 후보가 되고, 자기 자신은 빠지며, 빈 집합은 나오지 않는다.
- 순위가 추정 Jaccard 내림차순이고 `top_n`과 `min_similarity`가 동작한다.
- 인덱스에 없는 시그니처로 검색한 결과가 인덱스 안의 같은 시그니처로 검색한 결과와 같다.
- Jaccard를 정해 둔 합성 집합 쌍에서 후보가 되는 비율이 S-curve 이론값의 4 표준오차 안에 있다.

실제 데이터에서도 무작위 레시피 300개의 후보가 정의대로 전수 계산한 결과와 모두 같았다.

## 6. 한계

- 후보가 되는지는 확률적이다. 문턱 근처의 유사한 레시피는 놓칠 수 있다. 얼마나 놓치는지는 LSH 평가에서 정확한 Jaccard와 비교한다.
- 순위에 쓰는 값은 추정값이므로, 정확한 Jaccard로 매긴 순위와 다를 수 있다.
- 흔한 재료 때문에 큰 버킷이 생겨 일부 레시피의 조회가 느려진다.

## 참고 문헌

[1] J. Leskovec, A. Rajaraman, and J. D. Ullman, *Mining of Massive Datasets*, 3rd ed. Cambridge University Press, 2020, sec. 3.4.
