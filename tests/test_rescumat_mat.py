from __future__ import annotations

from pathlib import Path

import numpy as np

import _op2c

REPO_ROOT = Path(__file__).resolve().parents[5]
C_MAT = REPO_ROOT / "rescumat-2.8.0" / "PotentialData" / "C_AtomicData.mat"


def test_atomic_radials_reads_rescumat_mat() -> None:
    orbitals = _op2c.AtomicRadials()
    orbitals.build(str(C_MAT), 0, 0, 0)

    assert orbitals.symbol == "C"
    assert orbitals.nchi == 5
    assert orbitals.nphi == 13
    assert orbitals.lmax == 2
    assert orbitals.nzeta(0) == 2
    assert orbitals.nzeta(1) == 2
    assert orbitals.nzeta(2) == 1


def test_atom_pseudo_reads_rescumat_mat_projectors() -> None:
    pseudo = _op2c.Atom_pseudo()
    pseudo.init_from_upf(str(C_MAT), "mat", 10.0, False, 0.0)

    assert pseudo.psd == "C"
    assert pseudo.zv == 4.0
    assert pseudo.nbeta == 4
    assert pseudo.nproj == 4
    assert pseudo.beta_radials.nchi == 4
    assert pseudo.beta_radials.nphi == 16
    assert np.asarray(pseudo.d_real).shape == (17, 17)


def test_op2c_file_constructor_reads_rescumat_mat() -> None:
    op2c = _op2c.Op2c(
        1,
        1,
        False,
        f"{C_MAT.parent}/",
        [C_MAT.name],
        f"{C_MAT.parent}/",
        [C_MAT.name],
        "",
        0,
    )

    assert op2c.tcbd.orb.nphi(0) == 13
    assert op2c.tcbd.beta.nphi(0) == 16

    overlap = np.asarray(op2c.overlap(0, 0, [2.0, 0.0, 0.0], False))
    assert overlap.shape == (169,)
    assert np.linalg.norm(overlap) > 0.0

    ob, oxb, oyb, ozb = op2c.orb_r_beta([0], 0, [[0.0, 0.0, 0.0]], [0.5, 0.0, 0.0], False)
    assert len(ob) == 1
    assert np.asarray(ob[0]).shape == (13, 16)
    assert np.linalg.norm(np.asarray(ob[0])) > 0.0
    assert np.asarray(oxb[0]).shape == (13, 16)
    assert np.asarray(oyb[0]).shape == (13, 16)
    assert np.asarray(ozb[0]).shape == (13, 16)
