from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import patch

from serve import QueryRequest, SearchRequest, resolve_answer_result, resolve_search_result


def main() -> None:
    auth_context = SimpleNamespace(tenant_id="team_a", acl_groups=["ops"])

    search_request = SearchRequest(
        query="只看图片里的延迟指标",
        query_mode="multimodal",
        image_data_url="data:image/png;base64,ZmFrZQ==",
        history=["上一轮问题"],
        tenant_id="ignored",
        acl_groups=["ignored"],
        doc_ids=["dashboard-screenshot"],
        candidate_limit=7,
        context_limit=3,
        request_id="dispatch-search",
    )
    with (
        patch("serve.materialize_query_image", return_value="/tmp/query-image.png") as materialize,
        patch("serve.retrieve_multimodal", return_value="search-result") as retrieve,
    ):
        assert resolve_search_result(search_request, auth_context) == "search-result"
    materialize.assert_called_once_with(search_request)
    retrieve.assert_called_once()
    _, search_kwargs = retrieve.call_args
    assert search_kwargs["text_query"] == "只看图片里的延迟指标"
    assert search_kwargs["image_query_path"] == "/tmp/query-image.png"
    assert search_kwargs["tenant_id"] == "team_a"
    assert search_kwargs["acl_groups"] == ["ops"]
    assert search_kwargs["doc_ids"] == ["dashboard-screenshot"]

    answer_request = QueryRequest(
        query="解释这张图的召回率",
        query_mode="multimodal",
        image_data_url="data:image/png;base64,ZmFrZQ==",
        history=["上一轮问题"],
        doc_ids=["dashboard-screenshot"],
        candidate_limit=7,
        context_limit=3,
        request_id="dispatch-answer",
    )
    with (
        patch("serve.materialize_query_image", return_value="/tmp/query-image.png") as materialize,
        patch("serve.answer_multimodal_query", return_value="answer-result") as answer,
    ):
        assert resolve_answer_result(answer_request, auth_context) == "answer-result"
    materialize.assert_called_once_with(answer_request)
    answer.assert_called_once()
    _, answer_kwargs = answer.call_args
    assert answer_kwargs["text_query"] == "解释这张图的召回率"
    assert answer_kwargs["image_query_path"] == "/tmp/query-image.png"
    assert answer_kwargs["answer_query"] == "解释这张图的召回率"
    assert answer_kwargs["tenant_id"] == "team_a"
    assert answer_kwargs["acl_groups"] == ["ops"]
    assert answer_kwargs["doc_ids"] == ["dashboard-screenshot"]

    print("smoke_multimodal_dispatch=ok")


if __name__ == "__main__":
    main()
