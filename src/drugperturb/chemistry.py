"""Structure-only inputs for DrugPerturb; never fit transformations on responses.

Policy v1: RDKit Cleanup -> explicit counterion removal -> FragmentParent ->
Uncharger. Reject undefined atoms and unresolved mixtures *before* choosing a
parent. No tautomer canonicalization or inference from drug names is performed.
Counterion removal is a chemical identity convention, not a dose conversion.
Original SMILES, including any CXSMILES annotations, remain in every result.

Features intentionally omit stereochemistry (including enhanced/relative stereo).
Graphs also omit isotopes; source and canonical SMILES preserve audit information.
Scaffold separation does not guarantee low fingerprint similarity across splits.
"""
from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
import random
from typing import Any

import numpy as np
from rdkit import Chem, __version__ as RDKIT_VERSION
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.MolStandardize import rdMolStandardize
from rdkit.Chem.Scaffolds import MurckoScaffold

STANDARDIZATION_POLICY = 'rdkit_cleanup_explicit_counterions_fragmentparent_uncharger_v1'
ATOM_FEATURE_DIM = 10
BOND_FEATURE_DIM = 6

# Explicit identity-based list; arbitrary smaller organic components are NOT salts.
# The four organic acids cover mesylate/tosylate/citrate/tartrate forms in sci-Plex.
# Acid/base variants are compared after uncharging and without stereochemistry.
_COUNTERION_SMILES = (
    '[Na+]', '[K+]', '[Li+]', '[Ca+2]', '[Mg+2]', 'Cl', 'Br', 'I',
    '[Cl-]', '[Br-]', '[I-]', 'O', 'OP(=O)(O)O', 'OS(=O)(=O)O',
    'CS(=O)(=O)O', 'Cc1ccc(S(=O)(=O)O)cc1',
    'O=C(O)CC(O)(CC(=O)O)C(=O)O', 'O=C(O)C(O)C(O)C(=O)O',
)
_UNCHARGER = rdMolStandardize.Uncharger(canonicalOrder=True)
_COUNTERION_KEYS = frozenset(
    Chem.MolToSmiles(_UNCHARGER.uncharge(Chem.MolFromSmiles(s)), isomericSmiles=False)
    for s in _COUNTERION_SMILES
)
_MORGAN = rdFingerprintGenerator.GetMorganGenerator(
    radius=2, fpSize=2048, includeChirality=False,
)


def standardize_smiles(smiles: str) -> dict[str, Any]:
    """Return a traceable parent and scaffold; raise ValueError for exclusions.

    Multiple active fragments, even identical ones or acid/anion pairs, are
    rejected because fragment multiplicity can change the meaning of dose.
    Acyclic parents use their nonisomeric canonical structure as the group key rather
    than one shared empty Murcko scaffold. Ring scaffold keys ignore stereo.
    """
    if not isinstance(smiles, str) or not smiles.strip():
        raise ValueError('invalid SMILES: empty or non-string input')
    mol = Chem.MolFromSmiles(smiles)
    if mol is None or not mol.GetNumAtoms():
        raise ValueError('invalid SMILES: could not parse molecule')
    if any(atom.GetAtomicNum() == 0 or atom.HasQuery() for atom in mol.GetAtoms()):
        raise ValueError('undefined or query atom in SMILES')
    n_original_fragments = len(Chem.GetMolFrags(mol))
    enhanced_stereo = bool(mol.GetStereoGroups())
    cleaned = rdMolStandardize.Cleanup(mol)
    fragments = Chem.GetMolFrags(cleaned, asMols=True, sanitizeFrags=True)
    active = []
    removed = []
    for fragment in fragments:
        key = Chem.MolToSmiles(_UNCHARGER.uncharge(fragment), isomericSmiles=False)
        if key in _COUNTERION_KEYS:
            removed.append(Chem.MolToSmiles(fragment))
        else:
            active.append(fragment)
    if len(active) > 1:
        raise ValueError('unresolved active mixture: multiple non-counterion fragments')
    if not active:
        raise ValueError('no active parent remains after explicit counterion removal')
    parent = _UNCHARGER.uncharge(rdMolStandardize.FragmentParent(active[0]))
    canonical = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=True)
    scaffold = MurckoScaffold.GetScaffoldForMol(parent)
    scaffold_smiles = Chem.MolToSmiles(scaffold, canonical=True, isomericSmiles=False)
    acyclic_key = Chem.MolToSmiles(parent, canonical=True, isomericSmiles=False)
    group = f'murcko:{scaffold_smiles}' if scaffold_smiles else f'acyclic:{acyclic_key}'
    return {
        'original_smiles': smiles,
        'canonical_smiles': canonical,
        'scaffold_smiles': scaffold_smiles,
        'scaffold_group': group,
        'n_original_fragments': n_original_fragments,
        'removed_fragments': sorted(removed),
        'has_enhanced_stereo': enhanced_stereo,
        'standardization_policy': STANDARDIZATION_POLICY,
        'rdkit_version': RDKIT_VERSION,
    }


def standardize_compounds(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Standardize records with compound_id/smiles, preserving all input rows.

    Different IDs with the same parent remain separate records and receive the
    same group. Repeated IDs are allowed only when the canonical parent agrees;
    conflicting source labels raise rather than silently replacing response rows.
    """
    result = []
    seen = {}
    for record in records:
        compound_id = record['compound_id']
        if not isinstance(compound_id, str) or not compound_id.strip():
            raise ValueError('compound_id must be a nonempty string')
        standardized = standardize_smiles(record['smiles'])
        canonical = standardized['canonical_smiles']
        if compound_id in seen and seen[compound_id] != canonical:
            raise ValueError(f'conflicting structures for compound_id {compound_id!r}')
        seen[compound_id] = canonical
        result.append({**record, **standardized})
    return result


def split_by_structure(
    records: Iterable[Mapping[str, Any]], *, seed: int = 42,
    min_test_compounds: int = 20, min_test_groups: int = 10,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Split standardized records using structure only, approximately 70/15/15.

    Shuffle sorted scaffold keys once with seed=42. Reserve the two smallest
    groups for nonempty training/validation, then assign shuffled whole groups
    to test until its 15% target and feasible minimum are met. Validation takes
    approximately 15% from remaining groups; training receives the remainder.
    Large indivisible groups can force substantial deviations from these targets.
    No seed search, expression inspection, or response-dependent selection occurs.

    Counts distinguish source rows, compound IDs, unique canonical parents and
    groups. Minimum test compounds means *unique canonical parents*, so salt
    aliases cannot make a scientifically small test set appear large enough.
    """
    if min_test_compounds < 1 or min_test_groups < 1:
        raise ValueError('test minimum counts must be positive')
    rows = [dict(r) for r in records]
    structures: dict[str, set[str]] = defaultdict(set)
    parent_groups = {}
    id_parents = {}
    for row in rows:
        canonical, group = row['canonical_smiles'], row['scaffold_group']
        if canonical in parent_groups and parent_groups[canonical] != group:
            raise ValueError('canonical duplicate assigned conflicting scaffold groups')
        compound_id = row['compound_id']
        if compound_id in id_parents and id_parents[compound_id] != canonical:
            raise ValueError('compound_id has conflicting canonical structures')
        parent_groups[canonical] = group
        id_parents[compound_id] = canonical
        structures[group].add(canonical)
    if len(structures) < 3:
        raise ValueError('at least three structure groups are required for three splits')
    keys = sorted(structures)
    random.Random(seed).shuffle(keys)
    sizes = {key: len(structures[key]) for key in keys}
    n_compounds = len(parent_groups)
    # Reserve smallest groups, resolving equal sizes with the fixed shuffled order.
    reserved = sorted(keys, key=lambda key: sizes[key])[:2]
    available = [key for key in keys if key not in reserved]
    feasible = (
        len(available) >= min_test_groups
        and sum(sizes[key] for key in available) >= min_test_compounds
    )
    test_target = max(1, round(0.15 * n_compounds))
    required_compounds = min_test_compounds if feasible else 1
    required_groups = min_test_groups if feasible else 1
    test = []
    test_size = 0
    for key in available:
        test.append(key)
        test_size += sizes[key]
        if test_size >= max(test_target, required_compounds) and len(test) >= required_groups:
            break
    remaining = [key for key in keys if key not in set(test)]
    validation = []
    validation_size = 0
    for key in remaining[:-1]:
        validation.append(key)
        validation_size += sizes[key]
        if validation_size >= max(1, round(0.15 * n_compounds)):
            break
    assignments = {key: 'train' for key in keys}
    assignments.update({key: 'validation' for key in validation})
    assignments.update({key: 'test' for key in test})
    output = [{**row, 'split': assignments[row['scaffold_group']]} for row in rows]
    report: dict[str, Any] = {
        'seed': seed,
        'test_minimum_feasible': feasible,
        'test_minimum_met': test_size >= min_test_compounds and len(test) >= min_test_groups,
        'required_test_compounds': min_test_compounds,
        'required_test_groups': min_test_groups,
        'compound_count_unit': 'unique_canonical_parent',
        'acyclic_group_policy': 'nonisomeric_canonical_parent',
    }
    for split in ('train', 'validation', 'test'):
        selected = [r for r in output if r['split'] == split]
        report[split] = {
            'n_records': len(selected),
            'n_compound_ids': len({r['compound_id'] for r in selected}),
            'n_compounds': len({r['canonical_smiles'] for r in selected}),
            'n_groups': len({r['scaffold_group'] for r in selected}),
        }
    return output, report


def morgan_fingerprint(smiles: str) -> np.ndarray:
    """Binary radius-2 Morgan fingerprint, 2048 float32 entries, no chirality.

    Parent standardization is identical to the graph input and split identity.
    This is a fixed structural transform, so it does not learn from any split.
    """
    canonical = standardize_smiles(smiles)['canonical_smiles']
    mol = Chem.MolFromSmiles(canonical)
    return _MORGAN.GetFingerprintAsNumPy(mol).astype(np.float32)


def molecular_graph(smiles: str) -> dict[str, np.ndarray]:
    """Return NumPy graph arrays in canonical-parent atom order, without torch.

    x[N,10] float32 columns: atomic_number/100, degree/4, formal_charge/4,
    total_H/4, aromatic, in_ring, and one-hot hybridization [SP,SP2,SP3,other].
    edge_index[2,2B] int64 has both directions for each bond; no self loops.
    edge_attr[2B,6] float32 columns: single,double,triple,aromatic,conjugated,ring.
    Every bond's two directed copies have identical attributes. No-bond graphs
    retain (2,0) / (0,6) dimensions. Stereo/isotope information is not encoded.
    """
    canonical = standardize_smiles(smiles)['canonical_smiles']
    mol = Chem.MolFromSmiles(canonical)
    hybridizations = (Chem.HybridizationType.SP, Chem.HybridizationType.SP2, Chem.HybridizationType.SP3)
    atom_features = []
    for atom in mol.GetAtoms():
        hybrid = atom.GetHybridization()
        atom_features.append([
            atom.GetAtomicNum() / 100, atom.GetDegree() / 4,
            atom.GetFormalCharge() / 4, atom.GetTotalNumHs() / 4,
            float(atom.GetIsAromatic()), float(atom.IsInRing()),
            *(float(hybrid == h) for h in hybridizations),
            float(hybrid not in hybridizations),
        ])
    edges, bond_features = [], []
    for bond in mol.GetBonds():
        i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        bond_type = bond.GetBondType()
        feature = [
            float(bond_type == Chem.BondType.SINGLE),
            float(bond_type == Chem.BondType.DOUBLE),
            float(bond_type == Chem.BondType.TRIPLE),
            float(bond.GetIsAromatic()), float(bond.GetIsConjugated()), float(bond.IsInRing()),
        ]
        edges.extend(((i, j), (j, i)))
        bond_features.extend((feature, feature))
    return {
        'x': np.asarray(atom_features, dtype=np.float32).reshape(-1, ATOM_FEATURE_DIM),
        'edge_index': np.asarray(edges, dtype=np.int64).reshape(-1, 2).T,
        'edge_attr': np.asarray(bond_features, dtype=np.float32).reshape(-1, BOND_FEATURE_DIM),
    }
