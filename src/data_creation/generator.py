import os
import sys
import yaml
import time
from pathlib import Path
from src.utils.logging import get_logger

logger = get_logger(__name__)


class DataGenerator:
    """
    Wrapper for the AMLGentex data generation pipeline.

    For optimization/tuning workflows:
    - Spatial simulation (graph generation) is run once or reused
    - Temporal simulation can be run multiple times with different parameters
    """

    def __init__(self, conf_file: str):
        """
        Args:
            conf_file: Absolute path to the data configuration YAML file
        """
        if not os.path.isabs(conf_file):
            raise ValueError(f'conf_file must be an absolute path, got: {conf_file}')

        if not os.path.exists(conf_file):
            raise FileNotFoundError(f'Config file not found: {conf_file}')

        self.conf_file = conf_file

        # Load YAML config
        with open(conf_file, 'r') as f:
            self.config = yaml.safe_load(f)

        # Get directories
        self.data_creation_dir = Path(__file__).parent

        # Get project root (parent of src/)
        self.project_root = self.data_creation_dir.parent.parent

        # Convert relative paths in config to absolute paths
        self._make_paths_absolute()

        # Add data_creation directory to path for imports
        if str(self.data_creation_dir) not in sys.path:
            sys.path.insert(0, str(self.data_creation_dir))

    def _make_paths_absolute(self):
        """Convert relative paths in config to absolute paths relative to project root"""
        # Input directory
        if not os.path.isabs(self.config['input']['directory']):
            self.config['input']['directory'] = str(self.project_root / self.config['input']['directory'])

        # Temporal (spatial output) directory
        if not os.path.isabs(self.config['spatial']['directory']):
            self.config['spatial']['directory'] = str(self.project_root / self.config['spatial']['directory'])

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

        degree_file = self.config['input']['degree']

        # Get spatial output directory from config (already absolute)
        spatial_output = Path(self.config['spatial']['directory'])

        # Check if spatial outputs already exist
        if spatial_output.exists():
            spatial_files = list(spatial_output.glob('*.csv'))
            if spatial_files and not force:
                logger.info(f"Spatial simulation outputs found: {spatial_output}")
                logger.info(f"  Found {len(spatial_files)} CSV files")
                logger.info("Skipping spatial simulation (use force=True to regenerate)")
                return str(spatial_output)

        logger.info("Running spatial simulation...")

        # Save original working directory and argv
        orig_cwd = os.getcwd()
        orig_argv = sys.argv.copy()

        try:
            # Change to data_creation directory for simulation
            os.chdir(self.data_creation_dir)

            # Step 1: Generate degree distribution if needed (goes to spatial output)
            degree_path = spatial_output / degree_file
            if not degree_path.exists():
                logger.info(f"  [1/2] Generating degree distribution...")
                start = time.time()
                # Call directly with config dict (has absolute paths)
                generate_scalefree.generate_degree_file_from_config(self.config)
                logger.info(f"        Complete ({time.time() - start:.2f}s)")
            else:
                logger.info(f"  [1/2] Degree distribution found: {degree_path}")

            # Step 2: Generate transaction graph
            logger.info(f"  [2/2] Generating transaction graph...")
            start = time.time()
            # Call directly with config dict (has absolute paths)
            txgraph.generate_transaction_graph_from_config(self.config)
            logger.info(f"        Complete ({time.time() - start:.2f}s)")

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

        logger.info("Running temporal simulation...")

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
            simulator = AMLSimulator(self.config)
            simulator.load_accounts()
            simulator.load_transactions()
            simulator.load_normal_models()
            simulator.load_alert_members()

            start = time.time()
            simulator.run()
            elapsed = time.time() - start

            # Write output
            simulator.write_output()

            logger.info(f"  Complete: {len(simulator.transactions):,} transactions in {elapsed:.2f}s")

            # Construct tx_log path
            tx_log_file = self.config['output']['transaction_log']
            tx_log_path = temporal_output / tx_log_file

            if not tx_log_path.exists():
                raise FileNotFoundError(f'Transaction log not found: {tx_log_path}')

            return str(tx_log_path)

        finally:
            # Restore original working directory
            os.chdir(orig_cwd)

    def run_spatial_baseline(self, force=False):
        """
        Run spatial simulation Phase 1: Generate baseline and save checkpoint.

        Creates the transaction graph up to demographics assignment and saves a checkpoint.
        This baseline can then be used for multiple alert injection trials with different
        ML selector configurations via run_spatial_from_baseline().

        Args:
            force: If True, regenerate even if checkpoint exists

        Returns:
            Path to the saved checkpoint file
        """
        import spatial_simulation.generate_scalefree as generate_scalefree
        import spatial_simulation.transaction_graph_generator as txgraph

        sim_name = self.config['general']['simulation_name']
        input_dir = self.config['input']['directory']
        output_dir = self.config['spatial']['directory']  # spatial output
        degree_file = self.config['input']['degree']

        # Checkpoint path (in spatial output directory, not config)
        checkpoint_path = Path(output_dir) / 'baseline_checkpoint.pkl'

        # Check if checkpoint already exists
        if checkpoint_path.exists() and not force:
            logger.info(f"Baseline checkpoint found: {checkpoint_path}")
            logger.info("Skipping baseline generation (use force=True to regenerate)")
            return str(checkpoint_path)

        logger.info("Running spatial baseline generation (Phase 1)...")

        # Save original working directory and argv
        orig_cwd = os.getcwd()
        orig_argv = sys.argv.copy()

        try:
            # Change to data_creation directory for simulation
            os.chdir(self.data_creation_dir)

            # Step 1: Generate degree distribution if needed (goes to spatial output)
            degree_path = Path(output_dir) / degree_file
            if not degree_path.exists():
                logger.info(f"  [1/2] Generating degree distribution...")
                start = time.time()
                generate_scalefree.generate_degree_file_from_config(self.config)
                logger.info(f"        Complete ({time.time() - start:.2f}s)")
            else:
                logger.info(f"  [1/2] Degree distribution found: {degree_path}")

            # Step 2: Generate baseline (stops before alert injection)
            logger.info(f"  [2/2] Generating baseline graph...")
            start = time.time()
            saved_path = txgraph.generate_baseline(self.config, checkpoint_path=str(checkpoint_path))
            logger.info(f"        Complete ({time.time() - start:.2f}s)")
            logger.info(f"        Checkpoint saved to: {saved_path}")

        finally:
            # Restore original state
            os.chdir(orig_cwd)
            sys.argv = orig_argv

        return str(checkpoint_path)

    def run_spatial_from_baseline(self, checkpoint_path=None):
        """
        Run spatial simulation Phase 2: Inject alerts from baseline checkpoint.

        Loads a previously saved baseline and injects alert patterns using the ML selector
        configuration from the current config. This allows testing different ML selection
        parameters without regenerating the entire graph.

        Args:
            checkpoint_path: Path to baseline checkpoint. If None, uses default location.

        Returns:
            Path to the spatial simulation output directory
        """
        import spatial_simulation.transaction_graph_generator as txgraph

        output_dir = self.config['spatial']['directory']  # spatial output

        # Default checkpoint path (in spatial output directory, not config)
        if checkpoint_path is None:
            checkpoint_path = Path(output_dir) / 'baseline_checkpoint.pkl'

        if not Path(checkpoint_path).exists():
            raise FileNotFoundError(
                f"Baseline checkpoint not found: {checkpoint_path}\n"
                "Run run_spatial_baseline() first to generate the baseline."
            )

        # Get spatial output directory from config (already absolute)
        spatial_output = Path(self.config['spatial']['directory'])

        logger.info("Running alert injection from baseline (Phase 2)...")

        # Save original working directory
        orig_cwd = os.getcwd()

        try:
            # Change to data_creation directory for simulation
            os.chdir(self.data_creation_dir)

            start = time.time()
            txgraph.inject_alerts_from_baseline(
                self.config,
                checkpoint_path=str(checkpoint_path)
            )
            logger.info(f"        Complete ({time.time() - start:.2f}s)")

        finally:
            # Restore original state
            os.chdir(orig_cwd)

        return str(spatial_output)

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
