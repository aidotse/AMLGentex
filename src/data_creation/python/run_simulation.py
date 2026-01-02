#!/usr/bin/env python3
"""
AMLSim - Anti-Money Laundering Simulator
Main entry point for running complete simulation pipeline

Usage:
    python run_simulation.py <config_file>

Example:
    python run_simulation.py ../../experiments/100_accounts/data/param_files/conf.json
"""

import sys
import os
import json
import time
from pathlib import Path

# Change to project root directory
project_root = Path(__file__).parent.parent
os.chdir(project_root)

# Add python directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))

from spatial_simulation.generate_scalefree import main as generate_scalefree
from spatial_simulation.transaction_graph_generator import main as generate_graph
from spatial_simulation.insert_patterns import main as insert_patterns
from temporal_simulation.simulator import AMLSimulator


def run_pipeline(config_path):
    """Run complete AMLSim pipeline: spatial → temporal"""

    print("=" * 60)
    print("AMLSim - Anti-Money Laundering Simulator")
    print("=" * 60)

    # Load configuration
    with open(config_path, 'r') as f:
        config = json.load(f)

    sim_name = config['general']['simulation_name']
    input_dir = config['input']['directory']
    degree_file = config['input']['degree']

    print(f"\nSimulation: {sim_name}")
    print(f"Config: {config_path}")

    # Phase 1: Spatial Simulation (Graph Generation)
    print("\n" + "=" * 60)
    print("PHASE 1: SPATIAL SIMULATION (Graph Generation)")
    print("=" * 60)

    # Step 1.1: Generate degree distribution if needed
    degree_path = Path(input_dir) / degree_file
    if not degree_path.exists():
        print(f"\n[1/3] Generating degree distribution...")
        start = time.time()
        sys.argv = ['generate_scalefree.py', config_path]
        generate_scalefree()
        print(f"      ✓ Complete ({time.time() - start:.2f}s)")
    else:
        print(f"\n[1/3] Degree distribution found: {degree_path}")

    # Step 1.2: Generate transaction graph
    print(f"\n[2/3] Generating transaction graph...")
    start = time.time()
    sys.argv = ['transaction_graph_generator.py', config_path]
    generate_graph()
    print(f"      ✓ Complete ({time.time() - start:.2f}s)")

    # Step 1.3: Insert patterns if specified
    insert_patterns_file = config['input'].get('insert_patterns')
    if insert_patterns_file:
        patterns_path = Path(input_dir) / insert_patterns_file
        if patterns_path.exists():
            print(f"\n[3/3] Inserting patterns from {insert_patterns_file}...")
            start = time.time()
            sys.argv = ['insert_patterns.py', config_path]
            insert_patterns()
            print(f"      ✓ Complete ({time.time() - start:.2f}s)")
    else:
        print(f"\n[3/3] No patterns to insert")

    # Phase 2: Temporal Simulation (Time-Step Execution)
    print("\n" + "=" * 60)
    print("PHASE 2: TEMPORAL SIMULATION (Time-Step Execution)")
    print("=" * 60)

    print(f"\nInitializing simulator...")
    simulator = AMLSimulator(config_path)

    print(f"Loading accounts...")
    simulator.load_accounts()

    print(f"Loading transactions...")
    simulator.load_transactions()

    print(f"Loading normal models...")
    simulator.load_normal_models()

    print(f"Loading alert members...")
    simulator.load_alert_members()

    print(f"\nRunning temporal simulation...")
    start = time.time()
    simulator.run()
    elapsed = time.time() - start

    # Write output files
    print(f"\nWriting output files...")
    simulator.write_output()

    # Summary
    print("\n" + "=" * 60)
    print("SIMULATION COMPLETE")
    print("=" * 60)
    print(f"Total transactions: {len(simulator.transactions):,}")
    print(f"Temporal simulation time: {elapsed:.2f}s")
    print(f"Throughput: {len(simulator.transactions)/elapsed:,.0f} tx/s")
    print(f"\nOutputs saved to: {config['output']['directory']}")
    print("=" * 60)


def main():
    if len(sys.argv) != 2:
        print("Usage: python run_simulation.py <config_file>")
        print("\nExample:")
        print("  python run_simulation.py ../../experiments/100_accounts/data/param_files/conf.json")
        sys.exit(1)

    config_path = sys.argv[1]

    # Note: Working directory is already changed to project root at import time
    if not os.path.exists(config_path):
        print(f"Error: Config file not found: {config_path}")
        print(f"Current working directory: {os.getcwd()}")
        print(f"Note: Path should be relative to project root")
        sys.exit(1)

    run_pipeline(config_path)


if __name__ == '__main__':
    main()
