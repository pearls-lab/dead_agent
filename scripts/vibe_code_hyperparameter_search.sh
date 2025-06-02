#!/usr/bin/env python3

import subprocess
import os
import time
from pathlib import Path

class ExperimentLauncher:
    def __init__(self, conda_env="base", work_dir=".", log_base_dir="./logs/two_wall"):
        self.conda_env = conda_env
        self.work_dir = Path(work_dir).resolve()
        self.log_base_dir = Path(log_base_dir)
        
        # Create log directory
        self.log_base_dir.mkdir(parents=True, exist_ok=True)
        
        # Base command parameters
        self.base_params = {
            "env": "gridworld",
            "layers": 2,
            "parameters": 512,
            "root_checkpoint_save_dir": "./logs/",
            "gradient_steps": 6,
            "reward_dict": "./two_wall.json",
            "ent_coef": 0,
            "vf_coef": 0
        }
    
    def run_experiment(self, session_name, params, log_file):
        """Launch a single experiment in a tmux session"""
        
        # Build the python command
        cmd_parts = ["python", "train.py"]
        for key, value in params.items():
            cmd_parts.extend([f"--{key}", str(value)])
        
        python_cmd = " ".join(cmd_parts)
        
        # Create the full bash command that will run in tmux
        bash_cmd = f"""
        set -e
        cd {self.work_dir}
        source $(conda info --base)/etc/profile.d/conda.sh
        conda activate {self.conda_env}
        echo "Starting experiment: {session_name}"
        echo "Working directory: $(pwd)"
        echo "Conda environment: $CONDA_DEFAULT_ENV"
        echo "Command: {python_cmd}"
        echo "---"
        {python_cmd}
        """
        
        print(f"Starting experiment: {session_name}")
        print(f"Log file: {log_file}")
        print(f"Command: {python_cmd}")
        print("---")
        
        try:
            # Create tmux session
            subprocess.run([
                "tmux", "new-session", "-d", "-s", session_name,
                "bash", "-c", f"{bash_cmd} > {log_file} 2>&1"
            ], check=True)
            
            print(f"✓ Successfully launched {session_name}")
            
        except subprocess.CalledProcessError as e:
            print(f"✗ Failed to launch {session_name}: {e}")
        
        time.sleep(1)  # Small delay between launches
    
    def launch_all_experiments(self):
        """Launch all predefined experiments"""
        
        # Experiment configurations
        experiments = [
            # Learning rate experiments
            {
                "name": "lr_1e-05",
                "params": {**self.base_params, "algo": "ppo", "lr": 1e-05, "batch_size": 256, "script_id": "lr_1e-05"}
            },
            {
                "name": "lr_1e-04", 
                "params": {**self.base_params, "algo": "ppo", "lr": 1e-04, "batch_size": 256, "script_id": "lr_1e-04"}
            },
            {
                "name": "lr_1e-03",
                "params": {**self.base_params, "algo": "ppo", "lr": 1e-03, "batch_size": 256, "script_id": "lr_1e-03"}
            },
            {
                "name": "lr_3e-04",
                "params": {**self.base_params, "algo": "ppo", "lr": 3e-04, "batch_size": 256, "script_id": "lr_3e-04"}
            },
            
            # Algorithm experiments
            {
                "name": "algo_ppo",
                "params": {**self.base_params, "algo": "ppo", "lr": 1e-04, "batch_size": 256, "script_id": "algo_ppo"}
            },
            {
                "name": "algo_dqn",
                "params": {**self.base_params, "algo": "dqn", "lr": 1e-04, "batch_size": 256, "script_id": "algo_dqn"}
            },
            {
                "name": "algo_a2c",
                "params": {**self.base_params, "algo": "a2c", "lr": 1e-04, "batch_size": 256, "script_id": "algo_a2c"}
            },
            
            # Batch size experiments
            {
                "name": "batch_128",
                "params": {**self.base_params, "algo": "ppo", "lr": 1e-04, "batch_size": 128, "script_id": "batch_128"}
            },
            {
                "name": "batch_512",
                "params": {**self.base_params, "algo": "ppo", "lr": 1e-04, "batch_size": 512, "script_id": "batch_512"}
            },
            
            # Combined experiments
            {
                "name": "combined_high_lr_large_batch",
                "params": {**self.base_params, "algo": "ppo", "lr": 1e-03, "batch_size": 512, "script_id": "combined_1"}
            },
            {
                "name": "combined_dqn_low_lr",
                "params": {**self.base_params, "algo": "dqn", "lr": 1e-05, "batch_size": 128, "script_id": "combined_2"}
            }
        ]
        
        print(f"Launching {len(experiments)} experiments...")
        print(f"Conda environment: {self.conda_env}")
        print(f"Working directory: {self.work_dir}")
        print(f"Log directory: {self.log_base_dir}")
        print("=" * 50)
        
        for i, exp in enumerate(experiments, 1):
            session_name = f"exp_{exp['name']}"
            log_file = self.log_base_dir / f"{exp['name']}.txt"
            
            print(f"[{i}/{len(experiments)}] ", end="")
            self.run_experiment(session_name, exp["params"], log_file)
        
        print("\n" + "=" * 50)
        print("All experiments launched!")
        self.show_status()
    
    def show_status(self):
        """Show status of running experiments"""
        print("\n=== Experiment Status ===")
        try:
            result = subprocess.run(["tmux", "list-sessions"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                sessions = [line for line in result.stdout.split('\n') 
                           if line.startswith('exp_')]
                if sessions:
                    print("Running experiments:")
                    for session in sessions:
                        print(f"  {session}")
                else:
                    print("No experiments currently running")
            else:
                print("No tmux sessions found")
        except Exception as e:
            print(f"Error checking tmux sessions: {e}")
        
        print(f"\nLog files in {self.log_base_dir}:")
        try:
            log_files = list(self.log_base_dir.glob("*.txt"))
            if log_files:
                for log_file in sorted(log_files):
                    size = log_file.stat().st_size
                    print(f"  {log_file.name} ({size} bytes)")
            else:
                print("  No log files found")
        except Exception as e:
            print(f"Error listing log files: {e}")
    
    def cleanup(self):
        """Kill all experiment sessions"""
        print("Cleaning up experiment sessions...")
        try:
            result = subprocess.run(["tmux", "list-sessions"], 
                                  capture_output=True, text=True)
            if result.returncode == 0:
                sessions = [line.split(':')[0] for line in result.stdout.split('\n') 
                           if line.startswith('exp_')]
                
                for session in sessions:
                    try:
                        subprocess.run(["tmux", "kill-session", "-t", session], 
                                     check=True)
                        print(f"✓ Killed session: {session}")
                    except subprocess.CalledProcessError:
                        print(f"✗ Failed to kill session: {session}")
                
                if not sessions:
                    print("No experiment sessions to clean up")
            else:
                print("No tmux sessions found")
        except Exception as e:
            print(f"Error during cleanup: {e}")


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Launch ML training experiments in tmux sessions")
    parser.add_argument("--conda-env", default="base", help="Conda environment to activate")
    parser.add_argument("--work-dir", default=".", help="Working directory for experiments")
    parser.add_argument("--log-dir", default="./logs/two_wall", help="Directory for log files")
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Launch command
    subparsers.add_parser("launch", help="Launch all experiments")
    
    # Status command
    subparsers.add_parser("status", help="Show experiment status")
    
    # Cleanup command
    subparsers.add_parser("cleanup", help="Kill all experiment sessions")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    launcher = ExperimentLauncher(
        conda_env=args.conda_env,
        work_dir=args.work_dir,
        log_base_dir=args.log_dir
    )
    
    if args.command == "launch":
        launcher.launch_all_experiments()
    elif args.command == "status":
        launcher.show_status()
    elif args.command == "cleanup":
        launcher.cleanup()


if __name__ == "__main__":
    main()