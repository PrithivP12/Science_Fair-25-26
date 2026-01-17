"""
Data schema and QC rules for redox potential prediction.
"""
from typing import Dict, List, Optional, Tuple
import numpy as np

# ============================================================================
# FEATURE SCHEMA
# ============================================================================

# Required QC columns (must exist)
REQUIRED_QC_COLUMNS = [
    'ligand_protein_min_distance',
    'ligand_protein_clash_count',
    'cofactor_confidence_score',
]

# Critical features (reject if missing)
CRITICAL_FEATURES = [
    'hb_N5_nearest_dist',
    'hb_O4_nearest_dist',
    'net_charge_6A',
]

# Feature groups for analysis
FEATURE_GROUPS = {
    'esp': [c for c in [] if 'esp' in c.lower() or 'e_field' in c.lower()],
    'hbonds': [c for c in [] if c.startswith('hb_')],
    'charges': [c for c in [] if 'charge' in c.lower()],
    'geometry': [c for c in [] if any(x in c.lower() for x in ['distance', 'angle', 'stacking', 'burial', 'sasa'])],
    'composition': [c for c in [] if any(x in c.lower() for x in ['count', 'fraction', 'histidine', 'acidic', 'basic'])],
    'qc': ['ligand_protein_min_distance', 'ligand_protein_clash_count', 'cofactor_confidence_score'],
    'metadata': ['structure_source', 'has_waters', 'pdb_resolution_A', 'has_bfactors'],
}

# ============================================================================
# QC RULES
# ============================================================================

class QCRules:
    """Quality control rules for filtering samples."""
    
    # Distance thresholds
    MIN_DISTANCE_THRESHOLD = 1.2  # Å - minimum heavy-atom distance
    MAX_CLASH_COUNT = 10  # maximum number of clashes (distance <= 2.0 Å)
    
    # Confidence thresholds
    MIN_COFACTOR_CONFIDENCE = 0.3  # minimum cofactor confidence score
    
    # H-bond sanity checks
    MAX_HBOND_DISTANCE = 4.0  # Å - reject if H-bond distance > this
    MIN_HBOND_ANGLE = 90.0  # degrees - reject if angle < this (if available)
    
    # Resolution threshold (if available)
    MAX_RESOLUTION = 3.5  # Å - reject if resolution worse than this (optional)
    
    # Structure source preferences
    PREFERRED_SOURCE = 'holo_pdb'  # prefer experimental structures
    REJECT_SOURCE = None  # set to 'docked' to reject all docked structures
    
    @staticmethod
    def check_sample(row: Dict) -> Tuple[bool, List[str]]:
        """
        Check if a sample passes QC.
        
        Returns:
            (passes, reasons): tuple of (bool, list of rejection reasons)
        """
        reasons = []
        
        # Check min distance
        min_dist = row.get('ligand_protein_min_distance')
        if min_dist is not None and not np.isnan(min_dist):
            if min_dist < QCRules.MIN_DISTANCE_THRESHOLD:
                reasons.append(f"min_distance_too_low ({min_dist:.2f} < {QCRules.MIN_DISTANCE_THRESHOLD})")
        elif min_dist is None or np.isnan(min_dist):
            reasons.append("min_distance_missing")
        
        # Check clash count
        clash_count = row.get('ligand_protein_clash_count')
        if clash_count is not None and not np.isnan(clash_count):
            if clash_count > QCRules.MAX_CLASH_COUNT:
                reasons.append(f"clash_count_too_high ({clash_count} > {QCRules.MAX_CLASH_COUNT})")
        elif clash_count is None or np.isnan(clash_count):
            reasons.append("clash_count_missing")
        
        # Check cofactor confidence
        conf = row.get('cofactor_confidence_score')
        if conf is not None and not np.isnan(conf):
            if conf < QCRules.MIN_COFACTOR_CONFIDENCE:
                reasons.append(f"cofactor_confidence_too_low ({conf:.3f} < {QCRules.MIN_COFACTOR_CONFIDENCE})")
        elif conf is None or np.isnan(conf):
            reasons.append("cofactor_confidence_missing")
        
        # Check H-bond distances (sanity)
        for site in ['N5', 'O4', 'O2', 'N1']:
            hb_dist = row.get(f'hb_{site}_nearest_dist')
            if hb_dist is not None and not np.isnan(hb_dist):
                if hb_dist > QCRules.MAX_HBOND_DISTANCE:
                    reasons.append(f"hb_{site}_distance_too_large ({hb_dist:.2f} > {QCRules.MAX_HBOND_DISTANCE})")
        
        # Check structure source
        source = row.get('structure_source')
        if QCRules.REJECT_SOURCE and source == QCRules.REJECT_SOURCE:
            reasons.append(f"structure_source_rejected ({source})")
        
        # Check resolution (optional)
        resolution = row.get('pdb_resolution_A')
        if resolution is not None and not np.isnan(resolution):
            if resolution > QCRules.MAX_RESOLUTION:
                reasons.append(f"resolution_too_poor ({resolution:.2f} > {QCRules.MAX_RESOLUTION})")
        
        # Check critical features
        for feat in CRITICAL_FEATURES:
            val = row.get(feat)
            if val is None or (isinstance(val, float) and np.isnan(val)):
                reasons.append(f"critical_feature_missing ({feat})")
        
        passes = len(reasons) == 0
        return passes, reasons

# ============================================================================
# FEATURE TYPES
# ============================================================================

FEATURE_TYPES = {
    'continuous': [
        'ligand_protein_min_distance',
        'ligand_protein_clash_count',
        'cofactor_confidence_score',
        'net_charge_4A', 'net_charge_6A', 'net_charge_8A', 'net_charge_10A',
        'pocket_depth_proxy',
        'ligand_burial_fraction',
        'hb_N5_nearest_dist', 'hb_O4_nearest_dist', 'hb_O2_nearest_dist', 'hb_N1_nearest_dist',
        'hb_N5_nearest_angle', 'hb_O4_nearest_angle', 'hb_O2_nearest_angle', 'hb_N1_nearest_angle',
        'pdb_resolution_A',
    ],
    'categorical': [
        'structure_source',
        'cofactor_detected_from_complex',
        'cofactor_pred_from_sdf',
    ],
    'binary': [
        'has_waters',
        'has_bfactors',
    ],
    'count': [
        'hb_N5_count', 'hb_O4_count', 'hb_O2_count', 'hb_N1_count',
        'pocket_histidine_count_6A',
        'pocket_acidic_count_6A',
        'pocket_basic_count_6A',
    ],
}

# ============================================================================
# MISSING VALUE POLICY
# ============================================================================

MISSING_VALUE_POLICY = {
    'strategy': 'indicator',  # 'drop', 'mean', 'median', 'indicator', 'tree_friendly'
    'tree_friendly_threshold': 0.5,  # fraction of missing values to create indicator
    'min_samples_for_imputation': 10,  # minimum samples needed to impute
}

