import math
import pytest
from evaluation.score_retrieval import score_rows


def row(**kwargs):
    return dict(query_id='q1', relevant_ids=['a','b'], retrieved_ids=['x','a','b'], latency_ms=10, **kwargs)


def test_metrics_use_relevance_labels_not_internal_scores():
    result = score_rows([row()], k=2)['mean']
    assert result['recall_at_k'] == 0.5
    assert result['mrr_at_k'] == 0.5
    assert result['ndcg_at_k'] == pytest.approx((1/math.log2(3))/(1+1/math.log2(3)))


def test_no_results_counts_as_zero_not_skipped():
    missing = row()
    missing['query_id'] = 'q2'
    missing['retrieved_ids'] = []
    result = score_rows([row(), missing], k=3)
    assert result['queries'] == 2
    assert result['mean']['recall_at_k'] == 0.5


@pytest.mark.parametrize('change', [
    {'relevant_ids': []}, {'retrieved_ids': ['a','a']},
    {'latency_ms': float('nan')}, {'latency_ms': -1},
    {'query_id': ''}, {'retrieved_ids': 'a'},
])
def test_invalid_data_rejected(change):
    example = row()
    example.update(change)
    with pytest.raises(ValueError):
        score_rows([example])


def test_empty_and_duplicate_queries_rejected():
    with pytest.raises(ValueError):
        score_rows([])
    with pytest.raises(ValueError):
        score_rows([row(), row()])
