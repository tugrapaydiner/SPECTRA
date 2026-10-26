"""Compatibility imports for the public, solver-independent SBP1 verifier."""
from spectra.exact_proof import HEADER, decode_proof, verify_answer

__all__ = ["HEADER", "decode_proof", "verify_answer"]
