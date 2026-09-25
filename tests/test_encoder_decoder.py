import torch

from src.model import Encoder, Decoder, AutoEncoder


def test_encoder():

    # Sample input
    x = torch.randn(4, 10)

    encoder = Encoder()

    latent = encoder(x)

    # Check shape
    assert latent.shape == (4, 16)

    # Check for invalid values
    assert torch.isfinite(latent).all()


def test_decoder():

    # Sample latent representation
    z = torch.randn(4, 16)

    decoder = Decoder()

    output = decoder(z)

    # Check shape
    assert output.shape == (4, 10)

    # Check for invalid values
    assert torch.isfinite(output).all()


def test_encoder_decoder():

    # Sample input
    x = torch.randn(4, 10)

    model = AutoEncoder()

    output = model(x)

    # Output should have same shape as input
    assert output.shape == x.shape

    # Check for NaN / Inf
    assert torch.isfinite(output).all()