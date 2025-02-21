import numpy as np
import pytest

from diffusion.six_diffusion import SixDiffusion

@pytest.fixture
def diffusion():
    """
    Pytest fixture creating a SixDiffusion instance with default params.
    Adjust T, beta_start, etc. to match your usage.
    """
    return SixDiffusion(T=2000, beta_start=1e-6, beta_end=0.01)

def test_transform_ipv6_to_tokens_basic(diffusion):
    """
    Ensure transform_ipv6_to_tokens returns exactly 32 nybbles in [0..15].
    """
    ipv6_sample = "2001:0db8:85a3:0000:0000:8a2e:0370:7334"
    tokens = diffusion.transform_ipv6_to_tokens(ipv6_sample)

    assert len(tokens) == 32, f"Expected 32 tokens, got {len(tokens)}"
    for t in tokens:
        assert 0 <= t <= 15, f"Token {t} not in [0..15]."

def test_transform_ipv6_to_tokens_hex_cases(diffusion):
    """
    Test uppercase/lowercase hex parsing.
    """
    ipv6_sample = "ABCD:ef12:3456:7890:abcd:EF12:3456:7890"
    tokens = diffusion.transform_ipv6_to_tokens(ipv6_sample)

    # Should be length=32
    assert len(tokens) == 32

    # Spot-check known positions:
    # 'ABCD' => [10,11,12,13]
    assert tokens[0] == 10
    assert tokens[1] == 11
    assert tokens[2] == 12
    assert tokens[3] == 13

    # 'ef12' => [14,15,1,2]
    assert tokens[4] == 14
    assert tokens[5] == 15
    assert tokens[6] == 1
    assert tokens[7] == 2

def test_forward_diffusion_shape_dtype(diffusion):
    """
    forward_diffusion should return shape (32,) float32.
    """
    # Example tokens
    x0_tokens = np.array([0,1,2,3] + [15]*28, dtype=np.int64)
    t = diffusion.T - 1

    print("x0_tokens: ", x0_tokens)

    xt = diffusion.forward_diffusion(x0_tokens, t)

    print("xt_tokens: ", xt)

    assert xt.shape == (32,), f"Expected (32,) shape, got {xt.shape}"
    assert xt.dtype == np.float32, f"Expected float32, got {xt.dtype}"


def test_forward_diffusion_boundaries(diffusion):
    """
    - t=0 => nearly no noise => x_t ~ x0
    - t=T-1 => mostly noise => big difference from x0
    """
    x0_tokens = np.array([5]*32, dtype=np.int64)

    # t=0 => alpha_bar ~ near 1. x_t should be close to x0
    t0 = 0
    x_t0 = diffusion.forward_diffusion(x0_tokens, t0)
    diff0 = np.abs(x_t0 - x0_tokens.astype(np.float32)).mean()

    print("Diff at t=0: ", diff0)

    # Should be quite small
    assert diff0 < 0.5, f"For t=0, expected small diff, got mean diff={diff0:.4f}"

    # t=T-1 => alpha_bar might be small => mostly noise
    tmax = diffusion.T - 1
    x_tmax = diffusion.forward_diffusion(x0_tokens, tmax)
    diffmax = np.abs(x_tmax - x0_tokens.astype(np.float32)).mean()
    # Should be large

    print("Diff at t={tmax}: ", diffmax)

    assert diffmax > 2.0, f"For t=T-1, expected large diff, got mean diff={diffmax:.4f}"