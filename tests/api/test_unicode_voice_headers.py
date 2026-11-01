from app.api.v1.voice import _header_text
from starlette.responses import StreamingResponse


def test_unicode_voice_headers_are_latin1_safe() -> None:
    value = "😊 Hello — I’m Aiko! हिन्दी"
    encoded = _header_text(value)
    # Starlette encodes response headers using latin-1. This must not raise.
    response = StreamingResponse(iter([b"ok"]), headers={"X-Voice-Response": encoded})
    assert response.raw_headers
    assert encoded != value
