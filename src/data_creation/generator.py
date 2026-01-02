import os
import sys
import yaml
import time
from pathlib import Path


class DataGenerator:
    """
    Wrapper for the AMLGentex data generation pipeline.

    For optimization/tuning workflows:
    - Spatial simulation (graph generation) is run once or reused
    - Temporal simulation can be run multiple times with different parameters
    """

    def __init__(self, conf_file: str, verbose: bool = True):
        """
        Args:
            conf_file: Absolute path to the data configuration YAML file
            verbose: If True, print progress messages
        """
        if not os.path.isabs(conf_file):
            raise ValueError(f'conf_file must be an absolute path, got: {conf_file}')

        if not os.path.exists(conf_file):
            raise FileNotFoundError(f'Config file not found: {conf_file}')

        self.conf_file = conf_file
        self.verbose = verbose

        # Load YAML config
        with open(conf_file, 'r') as f:
            self.config = yaml.safe_load(f)

        # Get directories
        self.data_creation_dir = Path(__file__).parent
        self.python_dir = self.data_creation_dir / 'python'

        # Get project root (parent of src/)
        self.project_root = self.data_creation_dir.parent.parent

        # Convert relative paths in config to absolute paths
        self._make_paths_absolute()

        # Add python directory to path for imports
        if str(self.python_dir) not in sys.path:
            sys.path.insert(0, str(self.python_dir))

    def _make_paths_absolute(self):
        """Convert relative paths in config to absolute paths relative to project root"""
        # Input directory
        if not os.path.isabs(self.config['input']['directory']):
            self.config['input']['directory'] = str(self.project_root / self.config['input']['directory'])

        # Temporal (spatial output) directory
        if not os.path.isabs(self.config['temporal']['directory']):
            self.config['temporal']['directory'] = str(self.project_root / self.config['temporal']['directory'])

        # Output (temporal output) directory
        if not os.path.isabs(self.config['output']['directory']):
            self.config['output']['directory'] = str(self.project_root / self.config['output']['directory'])

    def run_spatial(self, force=False):
        """
        Run spatial simulation (graph generation).

        Args:
            force: If True, regenerate even if outputs exist

        Returns:
            Path to the spatial simulation output directory
        """
        import spatial_simulation.generate_scalefree as generate_scalefree
        import spatial_simulation.transaction_graph_generator as txgraph
        from spatial_simulation.insert_patterns import main as insert_patterns

        sim_name = self.config['general']['simulation_name']
        input_dir = self.config['input']['directory']
        degree_file = self.config['input']['degree']

        # Get spatial output directory from config (already absolute)
        spatial_output = Path(self.config['temporal']['directory'])

        # Check if spatial outputs already exist
        if spatial_output.exists():
            spatial_files = list(spatial_output.glob('*.csv'))
            if spatial_files and not force:
                if self.verbose:
                    print(f"Spatial simulation outputs found: {spatial_output}")
                    print(f"  Found {len(spatial_files)} CSV files")
                    print("Skipping spatial simulation (use force=True to regenerate)")
                return str(spatial_output)

        if self.verbose:
            print("Running spatial simulation...")

        # Save original working directory and argv
        orig_cwd = os.getcwd()
        orig_argv = sys.argv.copy()

        try:
            # Change to data_creation directory for simulation
            os.chdir(self.data_creation_dir)

            # Step 1: Generate degree distribution if needed
            degree_path = Path(input_dir) / degree_file
            if not degree_path.exists():
                if self.verbose:
                    print(f"  [1/3] Generating degree distribution...")
                start = time.time()
                # Call directly with config dict (has absolute paths)
                generate_scalefree.generate_degree_file_from_config(self.config)
                if self.verbose:
                    print(f"        Complete ({time.time() - start:.2f}s)")
            else:
                if self.verbose:
                    print(f"  [1/3] Degree distribution found: {degree_path}")

            # Step 2: Generate transaction graph
            if self.verbose:
                print(f"  [2/3] Generating transaction graph...")
            start = time.time()
            # Call directly with config dict (has absolute paths)
            txgraph.generate_transaction_graph_from_config(self.config)
            if self.verbose:
                print(f"        Complete ({time.time() - start:.2f}s)")

            # Step 3: Insert patterns if specified
            insert_patterns_file = self.config['input'].get('insert_patterns')
            if insert_patterns_file:
                patterns_path = Path(input_dir) / insert_patterns_file
                if patterns_path.exists():
                    if self.verbose:
                        print(f"  [3/3] Inserting patterns from {insert_patterns_file}...")
                    start = time.time()
                    sys.argv = ['insert_patterns.py', self.conf_file]
                    insert_patterns()
                    if self.verbose:
                        print(f"        Complete ({time.time() - start:.2f}s)")
            else:
                if self.verbose:
                    print(f"  [3/3] No patterns to insert")

        finally:
            # Restore original state
            os.chdir(orig_cwd)
            sys.argv = orig_argv

        return str(spatial_output)

    def run_temporal(self):
        """
        Run temporal simulation (time-step execution).

        Reloads config from file to pick up any parameter changes for optimization workflows.

        Returns:
            Path to the generated transaction log file
        """
        from temporal_simulation.simulator import AMLSimulator

        if self.verbose:
            print("Running temporal simulation...")

        # Reload config from file to pick up parameter changes
        with open(self.conf_file, 'r') as f:
            self.config = yaml.safe_load(f)
        self._make_paths_absolute()

        # Get temporal output directory from config (already absolute)
        temporal_output = Path(self.config['output']['directory'])

        # Save original working directory
        orig_cwd = os.getcwd()

        try:
            # Change to data_creation directory for simulation
            os.chdir(self.data_creation_dir)

            # Initialize and run simulator with config dict
            simulator = AMLSimulator(self.config, verbose=self.verbose)
            simulator.load_accounts()
            simulator.load_transactions()
            simulator.load_normal_models()
            simulator.load_alert_members()

            start = time.time()
            simulator.run()
            elapsed = time.time() - start

            # Write output
            simulator.write_output()

            if self.verbose:
                print(f"  Complete: {len(simulator.transactions):,} transactions in {elapsed:.2f}s")

            # Construct tx_log path
            tx_log_file = self.config['output']['transaction_log']
            tx_log_path = temporal_output / tx_log_file

            if not tx_log_path.exists():
                raise FileNotFoundError(f'Transaction log not found: {tx_log_path}')

            return str(tx_log_path)

        finally:
            # Restore original working directory
            os.chdir(orig_cwd)

    def __call__(self, spatial=True, force_spatial=False):
        """
        Run the complete simulation pipeline or just temporal simulation.

        Args:
            spatial: If True, run spatial simulation first (or check if exists)
            force_spatial: If True, force regeneration of spatial graph

        Returns:
            Path to the transaction log file
        """
        if spatial:
            self.run_spatial(force=force_spatial)

        tx_log_path = self.run_temporal()

        return tx_log_path
