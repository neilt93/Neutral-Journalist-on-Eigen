from __future__ import annotations

import asyncio
import base64
from unittest.mock import MagicMock

import httpx
import pytest

from src.attestation import onchain


class _FakeAsyncClient:
    def __init__(self, response):
        self._response = response

    async def __aenter__(self):
        return self

    async def __aexit__(self, exc_type, exc, tb):
        return False

    async def post(self, *_args, **_kwargs):
        return self._response


class TestTeeAttestationParsing:
    def test_parse_attestation_response_falls_back_to_raw_binary(self):
        response = MagicMock()
        response.content = b"\xab\xcd\xef"

        assert onchain._parse_attestation_response(response) == "abcdef"

    def test_parse_attestation_response_accepts_nested_base64_json(self):
        response = MagicMock()
        response.content = (
            b'{"result":{"attestation":"'
            + base64.b64encode(b"\xab\xcd\xef")
            + b'"}}'
        )

        assert onchain._parse_attestation_response(response) == "abcdef"

    def test_parse_attestation_response_uses_text_when_content_missing(self):
        response = MagicMock()
        response.content = b""
        response.text = "opaque-attestation"

        assert onchain._parse_attestation_response(response) == "6f70617175652d6174746573746174696f6e"

    def test_coerce_attestation_accepts_hex_and_byte_arrays(self):
        assert onchain._coerce_attestation_to_hex("0xabcdef") == "abcdef"
        assert onchain._coerce_attestation_to_hex([0xAB, 0xCD, 0xEF]) == "abcdef"

    def test_get_tee_attestation_endpoint_handles_binary_payload(self, monkeypatch):
        response = MagicMock()
        response.raise_for_status = MagicMock()
        response.content = b"\xab\xcd\xef"

        monkeypatch.setenv("EIGENCOMPUTE_TEE_ATTESTATION_ENDPOINT", "https://example.com/tee")
        monkeypatch.setattr(httpx, "AsyncClient", lambda *args, **kwargs: _FakeAsyncClient(response))

        attestation = asyncio.get_event_loop().run_until_complete(onchain.get_tee_attestation())

        assert attestation == "abcdef"
