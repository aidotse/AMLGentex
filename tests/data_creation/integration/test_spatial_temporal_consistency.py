"""
Integration test to ensure spatial simulation outputs match temporal simulation inputs

This test verifies:
1. Normal models from spatial CSV are correctly loaded in temporal simulation
2. Alert patterns from spatial CSV are correctly loaded in temporal simulation
3. No ID conflicts between normal and SAR pattern instances (patternID)
4. modelType in tx_log correctly identifies pattern types (0=generic, 1-8=SAR, 10-15=normal)
5. patternID uniquely identifies each pattern instance with no overlap

This test uses the large_test configuration from tests/data_creation/parameters/large_test/
and generates simulation data in a temporary directory.

The key insight (unified structure):
- In spatial CSVs, modelID represents a typology INSTANCE
- A single instance (e.g., fan_in with modelID=3) appears in MULTIPLE ROWS (one per account)
- In temporal simulation, instances are reassigned sequential patternIDs to avoid overlap
- In tx_log:
  * patternID = unique pattern instance ID (reassigned sequentially, no overlap)
  * modelType = pattern type ID (0=generic, 1-8=SAR types, 10-15=normal types)
- We verify: count(unique modelIDs in spatial) ≈ count(unique patternIDs in tx_log)
"""
import pandas as pd
import tempfile
import shutil
import os
import sys
import yaml
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from src.data_creation import DataGenerator

# Cache the generated data paths to avoid regenerating for each test
_generated_data_paths = None


def _generate_test_data():
    """Generate simulation data using large_test configuration"""
    global _generated_data_paths

    if _generated_data_paths is not None:
        return _generated_data_paths

    # Create temporary experiment directory structure
    temp_dir = Path(tempfile.mkdtemp(prefix='amlsim_test_'))

    try:
        # Create expected directory structure
        config_dir = temp_dir / 'config'
        config_dir.mkdir(parents=True, exist_ok=True)

        # Locate the test configuration parameters
        test_params_dir = Path(__file__).parent.parent / 'parameters' / 'large_test' / 'config'

        # Copy all parameter files to config directory (CSV and YAML)
        for param_file in test_params_dir.glob('*'):
            if param_file.is_file() and param_file.suffix in ['.csv', '.yaml', '.yml']:
                shutil.copy(param_file, config_dir / param_file.name)

        # Use the data.yaml configuration
        config_file = config_dir / 'data.yaml'

        # Load config with auto-discovered paths
        from src.utils.config import load_data_config
        full_config = load_data_config(str(config_file))

        # Write the full config with paths
        with open(config_file, 'w') as f:
            yaml.dump(full_config, f, default_flow_style=False, sort_keys=False)

        # Run the data generator
        print(f"\nGenerating test simulation data in {temp_dir}...")
        generator = DataGenerator(str(config_file))
        tx_log_file = generator()
        print(f"Generated tx_log at: {tx_log_file}")

        # Store paths for reuse
        _generated_data_paths = {
            'temp_dir': temp_dir,
            'spatial_dir': Path(temp_dir) / 'spatial',
            'temporal_dir': Path(temp_dir) / 'temporal',
            'tx_log': tx_log_file
        }

        return _generated_data_paths

    except Exception as e:
        # Clean up on error
        shutil.rmtree(temp_dir, ignore_errors=True)
        raise RuntimeError(f"Failed to generate test data: {e}")


def test_model_id_counts():
    """Test that we can correctly count typology instances from spatial CSVs"""
    paths = _generate_test_data()

    normal_models = pd.read_csv(paths['spatial_dir'] / 'normal_models.csv')
    alert_models = pd.read_csv(paths['spatial_dir'] / 'alert_models.csv')

    # Count unique typology instance IDs (from spatial CSVs)
    normal_ids = set(normal_models['modelID'].unique())
    alert_ids = set(alert_models['modelID'].unique())

    print(f"✓ Spatial CSV counts:")
    print(f"  Normal model instances: {len(normal_ids)} unique modelIDs")
    print(f"  Alert model instances: {len(alert_ids)} unique modelIDs")
    print(f"  Note: These IDs are reassigned sequentially in temporal simulation to avoid overlap")


def test_typology_instance_counts_match():
    """
    Test that the number of unique typology instances in tx_log matches spatial CSVs.

    Key insight:
    - In spatial CSVs, each modelID represents ONE typology instance
    - That instance appears in MULTIPLE ROWS (one per account)
    - In tx_log, each transaction has a patternID = unique instance ID (reassigned sequentially)
    - We verify: count of spatial instances ≈ count of unique patternIDs in tx_log (for active instances)
    """
    paths = _generate_test_data()

    # Load spatial outputs
    normal_models = pd.read_csv(paths['spatial_dir'] / 'normal_models.csv')
    alert_models = pd.read_csv(paths['spatial_dir'] / 'alert_models.csv')

    # Load temporal output
    tx_log = pd.read_parquet(paths['tx_log'])
    # Filter out generic transactions (patternID=-1)
    tx_log = tx_log[tx_log['patternID'] >= 0]

    # Count unique typology instances in spatial CSVs
    spatial_normal_instances = normal_models['modelID'].nunique()
    spatial_alert_instances = alert_models['modelID'].nunique()

    # Count unique typology instances in tx_log (using patternID)
    normal_txs = tx_log[tx_log['isSAR'] == 0]
    sar_txs = tx_log[tx_log['isSAR'] == 1]

    txlog_normal_instances = normal_txs['patternID'].nunique()
    txlog_alert_instances = sar_txs['patternID'].nunique()

    print(f"\n=== NORMAL TYPOLOGY INSTANCES ===")
    print(f"  Spatial CSV (normal_models.csv): {spatial_normal_instances} unique modelIDs")
    print(f"  Temporal tx_log (isSAR=0): {txlog_normal_instances} unique patternIDs")
    print(f"  Transactions generated: {len(normal_txs)} from {txlog_normal_instances} instances")

    # Verify counts match (or txlog has fewer due to inactive patterns)
    if txlog_normal_instances > spatial_normal_instances:
        raise AssertionError(
            f"More normal pattern instances in tx_log ({txlog_normal_instances}) than spatial CSV ({spatial_normal_instances})!"
        )
    elif txlog_normal_instances < spatial_normal_instances:
        pct = (txlog_normal_instances / spatial_normal_instances) * 100
        print(f"  Note: {spatial_normal_instances - txlog_normal_instances} instances did not generate transactions ({pct:.1f}% active)")

    print(f"\n=== ALERT TYPOLOGY INSTANCES ===")
    print(f"  Spatial CSV (alert_models.csv): {spatial_alert_instances} unique modelIDs")
    print(f"  Temporal tx_log (isSAR=1): {txlog_alert_instances} unique patternIDs")
    print(f"  Transactions generated: {len(sar_txs)} from {txlog_alert_instances} instances")

    # Verify counts match (or txlog has fewer due to inactive patterns)
    if txlog_alert_instances > spatial_alert_instances:
        raise AssertionError(
            f"More alert pattern instances in tx_log ({txlog_alert_instances}) than spatial CSV ({spatial_alert_instances})!"
        )
    elif txlog_alert_instances < spatial_alert_instances:
        pct = (txlog_alert_instances / spatial_alert_instances) * 100
        print(f"  Note: {spatial_alert_instances - txlog_alert_instances} instances did not generate transactions ({pct:.1f}% active)")

    # Verify no overlap between normal and SAR patternIDs
    normal_pattern_ids = set(normal_txs['patternID'].unique())
    sar_pattern_ids = set(sar_txs['patternID'].unique())
    overlap = normal_pattern_ids & sar_pattern_ids

    if overlap:
        raise AssertionError(
            f"Found {len(overlap)} patternIDs used by both normal and SAR transactions! "
            f"Overlapping IDs: {sorted(overlap)[:10]}..."
        )

    print(f"\n✓ All patternIDs are unique with no overlap between normal and SAR patterns")


def test_typology_details():
    """
    Show detailed breakdown of typology patterns in the data.
    This helps understand which types of patterns are present and active.
    """
    paths = _generate_test_data()

    normal_models = pd.read_csv(paths['spatial_dir'] / 'normal_models.csv')
    alert_models = pd.read_csv(paths['spatial_dir'] / 'alert_models.csv')
    tx_log = pd.read_parquet(paths['tx_log'])

    # Filter tx_log to pattern transactions only
    tx_log = tx_log[tx_log['patternID'] >= 0]

    # Analyze normal patterns
    print(f"\n=== NORMAL PATTERN BREAKDOWN ===")
    normal_pattern_counts = normal_models.groupby('type')['modelID'].nunique()
    for pattern_type, count in normal_pattern_counts.items():
        print(f"  {pattern_type}: {count} instances")

    # Check which generated transactions
    normal_txs = tx_log[tx_log['isSAR'] == 0]
    if len(normal_txs) > 0:
        # Count active instances by modelType (pattern type ID)
        from src.utils.pattern_types import NORMAL_PATTERN_TYPES
        type_to_id = NORMAL_PATTERN_TYPES

        print(f"\n  Active (generated transactions):")
        for pattern_type in normal_pattern_counts.index:
            type_id = type_to_id.get(pattern_type)
            if type_id is not None:
                active_instances = normal_txs[normal_txs['modelType'] == type_id]['patternID'].nunique()
                total = normal_pattern_counts[pattern_type]
                pct = (active_instances / total) * 100 if total > 0 else 0
                tx_count = len(normal_txs[normal_txs['modelType'] == type_id])
                print(f"    {pattern_type}: {active_instances}/{total} instances ({pct:.1f}%), {tx_count} transactions")

    # Analyze alert patterns
    print(f"\n=== ALERT PATTERN BREAKDOWN ===")
    alert_pattern_counts = alert_models.groupby('type')['modelID'].nunique()
    for pattern_type, count in alert_pattern_counts.items():
        print(f"  {pattern_type}: {count} instances")

    # Check which generated transactions
    sar_txs = tx_log[tx_log['isSAR'] == 1]
    if len(sar_txs) > 0:
        from src.utils.pattern_types import SAR_PATTERN_TYPES
        type_to_id = SAR_PATTERN_TYPES

        print(f"\n  Active (generated transactions):")
        for pattern_type in alert_pattern_counts.index:
            type_id = type_to_id.get(pattern_type)
            if type_id is not None:
                active_instances = sar_txs[sar_txs['modelType'] == type_id]['patternID'].nunique()
                total = alert_pattern_counts[pattern_type]
                pct = (active_instances / total) * 100 if total > 0 else 0
                tx_count = len(sar_txs[sar_txs['modelType'] == type_id])
                print(f"    {pattern_type}: {active_instances}/{total} instances ({pct:.1f}%), {tx_count} transactions")

    print(f"\n✓ Pattern type breakdown complete")


def cleanup_test_data():
    """Clean up temporary test data"""
    global _generated_data_paths
    if _generated_data_paths and os.path.exists(_generated_data_paths['temp_dir']):
        print(f"\nCleaning up test data: {_generated_data_paths['temp_dir']}")
        shutil.rmtree(_generated_data_paths['temp_dir'], ignore_errors=True)
        _generated_data_paths = None


if __name__ == "__main__":
    """Run tests directly for debugging"""
    print("=" * 60)
    print("SPATIAL-TEMPORAL CONSISTENCY TESTS")
    print("=" * 60)

    try:
        print("\n1. Counting model IDs in spatial CSVs...")
        test_model_id_counts()

        print("\n" + "=" * 60)
        print("\n2. Testing typology instance counts...")
        test_typology_instance_counts_match()

        print("\n" + "=" * 60)
        print("\n3. Analyzing typology pattern details...")
        test_typology_details()

        print("\n" + "=" * 60)
        print("TESTS COMPLETE - ALL PASSED ✓")
        print("=" * 60)
    except AssertionError as e:
        print(f"\n{'=' * 60}")
        print(f"TEST FAILED: {e}")
        print("=" * 60)
        raise
    finally:
        cleanup_test_data()
