from pathlib import Path


def test_m15_theory_claims_are_qualified_in_public_sources():
    spectral = Path("model/spectral.py").read_text().lower()
    architecture = Path("docs/ARCHITECTURE.md").read_text().lower()
    vq = Path("model/latent_vq.py").read_text().lower()
    energy = Path("model/energy.py").read_text().lower()

    assert "guarantee the latent trajectory stays" not in spectral
    assert "does not guarantee" in spectral
    assert "not a proof of dynamical isometry" in spectral

    assert "topology survives arbitrarily deep search" not in vq
    assert "do **not** imply" in vq

    assert "does **not** by itself identify out-of-distribution states" in energy

    # The blueprint must not present ensemble disagreement or sampled regularizers
    # as certificates of OOD detection, full rank, or dynamical isometry.
    assert "disagreement is a calibrated epistemic signal" not in architecture
    assert "ood latents incur a large `β σ` penalty and are not chased" not in architecture
    assert "preventing a rank-deficient state" not in architecture
    assert "which is zero iff `j` is an isometry" not in architecture
