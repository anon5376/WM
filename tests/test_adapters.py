import torch

from wm.adapters.byte_tokenizer import ByteTokenizer
from wm.adapters.modules import (
    ByteInverseHead,
    GridInverseHead,
    GridPatchEmbed,
    SignalFrameEmbed,
    SignalInverseHead,
    TextEventByteEmbedder,
    TypePositionEmbeddings,
)
from wm.data.grid import GRID_CHANNELS, VIEW_H, VIEW_W


def test_byte_tokenizer_round_trip_exact_text_and_bytes():
    tok = ByteTokenizer()
    text = "agent opens gate at 2 3 .\n"
    assert tok.decode(tok.encode(text)) == text
    payload = bytes(range(256))
    assert tok.decode_bytes(tok.encode_bytes(payload)) == payload


def test_text_event_embedder_and_inverse_shapes():
    d_model = 32
    ids = torch.tensor([[1, 2, 3, 255]])
    emb = TextEventByteEmbedder(d_model)(ids)
    assert emb.shape == (1, 4, d_model)
    logits = ByteInverseHead(d_model)(emb)
    assert logits.shape == (1, 4, 256)


def test_grid_patch_embed_and_inverse_shapes():
    d_model = 32
    frames = torch.zeros(2, 3, GRID_CHANNELS, VIEW_H, VIEW_W)
    emb = GridPatchEmbed(d_model)(frames)
    assert emb.shape == (2, 3, d_model)
    inv = GridInverseHead(d_model)(emb)
    assert inv.shape == frames.shape


def test_signal_frame_embed_and_inverse_shapes():
    d_model = 32
    frames = torch.zeros(2, 8, 2)
    emb = SignalFrameEmbed(d_model)(frames)
    assert emb.shape == (2, 8, d_model)
    inv = SignalInverseHead(d_model)(emb)
    assert inv.shape == frames.shape


def test_type_position_embeddings_are_shape_preserving():
    x = torch.zeros(2, 5, 32)
    y = TypePositionEmbeddings(32, max_len=8)(x, modality_id=1)
    assert y.shape == x.shape
    assert torch.count_nonzero(y) > 0

