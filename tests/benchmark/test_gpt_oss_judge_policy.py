from __future__ import annotations

from benchmark import judge_sova_outputs as judge

class FakeResponse:
    def raise_for_status(self):
        pass
    def json(self):
        return {"choices": [{"message": {"content": "{\"correct\": true, \"reason\": \"matches\"}"}}]}

def test_gpt_oss_judge_uses_groq_endpoint_not_openai(monkeypatch):
    seen = {}
    def fake_post(self, url, **kwargs):
        seen["url"] = url
        seen["body"] = kwargs["json"]
        seen["authorization"] = kwargs["headers"]["Authorization"]
        return FakeResponse()
    monkeypatch.setattr("httpx.Client.post", fake_post)
    import httpx
    with httpx.Client() as client:
        result = judge.call_judge(client, "openai/gpt-oss-120b", "fake-key", "judge this")
    assert seen["url"] == "https://api.groq.com/openai/v1/chat/completions"
    assert seen["body"]["model"] == "openai/gpt-oss-120b"
    assert seen["authorization"] == "Bearer fake-key"
    assert result["correct"] is True
