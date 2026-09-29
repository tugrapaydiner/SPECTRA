# Bind audit acceptance to the source that actually executed

`runner.py` requires an externally trusted raw SHA256 for an auditor. It verifies
that digest before executing the exact bytes read, not a later import of the file.
A successful receipt names both auditor and runner digests and embeds the audit's
own result. Existing output files are never overwritten. This blocks accidental
acceptance of an unverified source copy or reuse of an old PASS summary.

`seal.py` records every artifact file and checks exact closure: changed, missing,
additional and symlinked members fail. A trusted manifest digest is optional in the
API but should be supplied for delivery acceptance. Neither program is a sandbox,
cryptographic signature, statistical proof, or substitute for scientific review.
The auditor is trusted executable code. Its dependencies/records must be covered
by its own checks or by the artifact closure, not inferred from a single hash.

The prior SPECTRA PR40 source discrepancy cannot be resolved without its missing
local source copy. These tools prevent that failure for newly delivered evidence;
they do not retrospectively make old missing evidence valid.
