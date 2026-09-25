"""API 层测试：提交、查询、校验与原子拒绝。"""

import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("DATABASE_PATH", str(tmp_path / "test.db"))
    with TestClient(app) as c:
        yield c


def eco_request(**overrides):
    payload = {
        "sequence": "AAAAGAATTCTTTT",
        "topology": "linear",
        "enzymes": [{"name": "EcoRI", "pattern": "GAATTC", "cut_top": 1, "cut_bottom": 5}],
    }
    payload.update(overrides)
    return payload


def test_create_and_get_digest(client):
    resp = client.post("/digests", json=eco_request())
    assert resp.status_code == 201
    body = resp.json()
    assert body["id"] == 1
    assert body["topology"] == "linear"
    assert body["sequence_length"] == 14
    assert len(body["cuts"]) == 1
    assert body["cuts"][0]["end_type"] == "5prime"
    assert body["cuts"][0]["overhang"] == "AATT"
    assert [f["length"] for f in body["fragments"]] == [5, 9]

    got = client.get(f"/digests/{body['id']}")
    assert got.status_code == 200
    assert got.json() == body  # 查询结果与提交时完全一致


def test_list_digests(client):
    client.post("/digests", json=eco_request())
    client.post("/digests", json=eco_request(sequence="GCGCGCGC", topology="circular"))
    resp = client.get("/digests")
    assert resp.status_code == 200
    items = resp.json()
    assert len(items) == 2
    assert items[0]["id"] == 1 and items[0]["cut_count"] == 1
    assert items[1]["id"] == 2 and items[1]["cut_count"] == 0
    assert items[1]["fragment_count"] == 1  # 无切口保留整条分子


def test_get_missing_digest_returns_404(client):
    assert client.get("/digests/999").status_code == 404


def test_illegal_offset_rejects_entire_save(client):
    bad = eco_request(
        enzymes=[
            {"name": "Good", "pattern": "GAATTC", "cut_top": 1, "cut_bottom": 5},
            {"name": "Bad", "pattern": "GATATC", "cut_top": 7, "cut_bottom": 3},
        ]
    )
    resp = client.post("/digests", json=bad)
    assert resp.status_code == 422
    assert client.get("/digests").json() == []  # 整次保存被拒绝，无任何落库


def test_negative_offset_rejected(client):
    bad = eco_request(
        enzymes=[{"name": "Neg", "pattern": "GAATTC", "cut_top": -1, "cut_bottom": 5}]
    )
    assert client.post("/digests", json=bad).status_code == 422
    assert client.get("/digests").json() == []


def test_invalid_sequence_base_rejected(client):
    resp = client.post("/digests", json=eco_request(sequence="ACGTXACGT"))
    assert resp.status_code == 422
    assert client.get("/digests").json() == []


def test_invalid_pattern_base_rejected(client):
    bad = eco_request(
        enzymes=[{"name": "Bad", "pattern": "GAUZ", "cut_top": 1, "cut_bottom": 1}]
    )
    assert client.post("/digests", json=bad).status_code == 422


def test_empty_enzyme_list_rejected(client):
    assert client.post("/digests", json=eco_request(enzymes=[])).status_code == 422


def test_circular_digest_via_api(client):
    payload = eco_request(sequence="ATTCGGGGGA", topology="circular")
    resp = client.post("/digests", json=payload)
    assert resp.status_code == 201
    body = resp.json()
    assert body["cuts"][0]["sites"][0]["start"] == 8  # 跨原点命中
    assert len(body["fragments"]) == 1
    assert body["fragments"][0]["length"] == 10


def test_circular_shared_top_cut_counts_consistent(client):
    # 两个物理切口共用同一正链坐标：保存两个 cuts，片段按唯一边界生成
    payload = eco_request(
        sequence="AACAA",
        topology="circular",
        enzymes=[
            {"name": "E1", "pattern": "AC", "cut_top": 0, "cut_bottom": 0},
            {"name": "E2", "pattern": "AC", "cut_top": 0, "cut_bottom": 1},
        ],
    )
    resp = client.post("/digests", json=payload)
    assert resp.status_code == 201
    body = resp.json()
    assert len(body["cuts"]) == 2
    assert len(body["fragments"]) == 1  # 唯一边界开环为一条全长片段
    frag = body["fragments"][0]
    assert frag["length"] == 5
    assert frag["sequence"] == "ACAAA"
    assert frag["left_end"] == {"type": "blunt", "overhang": ""}

    got = client.get(f"/digests/{body['id']}")
    assert got.status_code == 200
    assert got.json() == body  # 明细与提交时完全一致

    items = client.get("/digests").json()
    assert len(items) == 1
    # 列表摘要计数与明细一致
    assert items[0]["cut_count"] == len(body["cuts"]) == 2
    assert items[0]["fragment_count"] == len(body["fragments"]) == 1


def test_circular_conflicting_same_coordinate_ends_rejected_and_not_persisted(client):
    # 环形 AAAA：偏移 (0,3) 的事件取模后落到 (0,3) 为 5' 突出 AAA，
    # 偏移 (1,0) 的事件同样落到 (0,3) 却为 3' 突出 A ——
    # 同一归一化坐标末端互不相容：422，detail 含冲突坐标，且不落库。
    payload = eco_request(
        sequence="AAAA",
        topology="circular",
        enzymes=[
            {"name": "A", "pattern": "AAAA", "cut_top": 0, "cut_bottom": 3},
            {"name": "B", "pattern": "AAAA", "cut_top": 1, "cut_bottom": 0},
        ],
    )
    resp = client.post("/digests", json=payload)
    assert resp.status_code == 422
    detail = resp.json()["detail"]
    # 全部四个取模后冲突坐标都出现在错误说明中
    for coord in ("(0, 3)", "(1, 0)", "(2, 1)", "(3, 2)"):
        assert coord in detail
    assert "5prime" in detail and "3prime" in detail
    assert client.get("/digests").json() == []  # 创建请求被拒绝，无任何落库

    # 交换酶顺序（同一物理输入）错误结果一致，不随酶名排序变化
    payload["enzymes"] = list(reversed(payload["enzymes"]))
    resp2 = client.post("/digests", json=payload)
    assert resp2.status_code == 422
    assert resp2.json()["detail"] == detail
    assert client.get("/digests").json() == []


def test_circular_consistent_same_coordinate_merged_via_api(client):
    # 同坐标、同平端、同突出端的合法事件继续合并
    payload = eco_request(
        sequence="AC",
        topology="circular",
        enzymes=[
            {"name": "E1", "pattern": "AC", "cut_top": 0, "cut_bottom": 0},
            {"name": "E2", "pattern": "AC", "cut_top": 0, "cut_bottom": 0},
        ],
    )
    resp = client.post("/digests", json=payload)
    assert resp.status_code == 201
    body = resp.json()
    assert len(body["cuts"]) == 1
    assert (body["cuts"][0]["top_cut"], body["cuts"][0]["bottom_cut"]) == (0, 0)
    assert body["cuts"][0]["enzymes"] == ["E1", "E2"]
    assert len(body["fragments"]) == 1  # 单边界开环为一条全长片段


def test_same_input_same_result_order(client):
    payload = eco_request(
        sequence="GAATTCAAAAGATATCTTTTGGTACC",
        topology="circular",
        enzymes=[
            {"name": "KpnI", "pattern": "GGTACC", "cut_top": 5, "cut_bottom": 1},
            {"name": "EcoRI", "pattern": "GAATTC", "cut_top": 1, "cut_bottom": 5},
            {"name": "EcoRV", "pattern": "GATATC", "cut_top": 3, "cut_bottom": 3},
        ],
    )
    r1 = client.post("/digests", json=payload).json()
    r2 = client.post("/digests", json=payload).json()
    assert r1["cuts"] == r2["cuts"]
    assert r1["fragments"] == r2["fragments"]
