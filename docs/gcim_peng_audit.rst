Zheng--Peng convention audit
===========================

The molecular and saved-Hamiltonian GCIM interfaces use
exp(theta*A) with theta=pi/4 by default. Earlier releases applied
theta/2; repeating an older trajectory therefore requires explicitly
passing theta=pi/8 to this release.

Overlap handling uses only spectral truncation, retaining eigenvalues above
1e-13. Numerically this is evaluated as squared singular values of the
state-column matrix. A diagonal overlap shift is not applied. A constant
energy origin is subtracted before projection and restored afterwards.

singlet_sd contains 54 normalized generators for CAS(6e,6o).
singlet_gsd contains the 330 normalized spin-scalar generalized generators
in the order and sign convention of the original QuGCM operator pool.
The older gsd option denotes a different, spin-projection-conserving pool;
it must not be labelled SingletGSD.

The gradient is evaluated on the cumulative unitary product state, not the
Ritz state. At each selection, both an individual-generator state and a
cumulative state are retained, with only one new state on the first step.
There are therefore 2K raw basis states after K selections.

For the published first-pass construction, use unique selection. Running the
complete pool deliberately disables the paper's early energy-change stopping
rule. The matrix API also provides explicitly labelled reuse extensions:
cycle refreshes eligibility but retains the reference, product, and basis;
repeat_after_first permits arbitrary reuse after a unique first pass.
The molecular API exposes the same choices through selection_policy.
For five CAS(6e,6o) SD passes, use policy="cycle", max_iterations=270
in the matrix API, or selection_policy="cycle", max_iterations=270 in
the molecular API. The repetitive comparison uses "repeat_after_first".

Selection 55 continues the product of the first 54 unitaries, appends two
columns to the existing 108-column basis, and diagonalizes that enlarged
subspace. No Ritz state becomes a new reference. Duplicate individual-state
columns are harmless: overlap truncation removes their redundant directions.
Calling GCIM again with the previous Ritz state is a different algorithm and
must not be used for these pool-refresh or repetitive calculations.

Reference: https://www.nature.com/articles/s41534-024-00916-8

Verification of this branch (6 October 2026)
-------------------------------------------

The deposited linear-H4 / STO-3G example at 2 bohr separation uses the
authors' 66-generator singlet_GSD pool. Independent selection reproduces
all 23 deposited operator indices; the largest energy difference is
1.34e-15 Ha. Chemical accuracy first occurs at selection 11 in both runs.
This verifies one published example, not all systems or the stopping rule.

For the deposited U2 CAS(6e,6o) Hamiltonian at 2.60 Angstrom, the 54-generator
SingletSD pool reaches 1.5936 mEh at selections 92 (cycle) and 95
(repeat_after_first). Both retain 108 raw columns at selection 54, 110 at
55, and 540 at 270, including every preceding column unchanged. Thresholding
acts on the overlap spectrum, not on the stored basis list. These reuse
extensions are not attributed to the original paper.

The runnable example is examples/gcim_pool_refresh.py. It accepts the saved
sector Hamiltonian, reference/exact-state archive and S-squared matrix.
The supplied U2 reproduction uses the unmodified 2.60-Angstrom inputs.
