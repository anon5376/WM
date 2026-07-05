from __future__ import annotations


class ByteTokenizer:
    vocab_size = 256

    def encode_bytes(self, payload: bytes) -> list[int]:
        return list(payload)

    def decode_bytes(self, ids: list[int]) -> bytes:
        return bytes(ids)

    def encode(self, text: str) -> list[int]:
        return self.encode_bytes(text.encode("utf-8"))

    def decode(self, ids: list[int]) -> str:
        return self.decode_bytes(ids).decode("utf-8")

