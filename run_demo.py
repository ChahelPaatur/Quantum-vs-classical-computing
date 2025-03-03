#!/usr/bin/env python3
"""
Demo script for running a simplified benchmark of quantum vs classical vs hybrid models.
"""
import os
import argparse
from benchmark import run_benchmark


def parse_args():
    """Parse command line arguments for the demo."""
    parser = argparse.ArgumentParser(description='Run a demo benchmark of quantum vs classical vs hybrid models')
    
    parser.add_argument('--task', type=str, default='classification', choices=['classification', 'regression'],
                        help='Type of machine learning task')
    
    parser.add_argument('--output_dir', type=str, default='demo_results',
                        help='Directory to save results')
    
    return parser.parse_args()


def run_demo():
    """Run a simplified benchmark for demonstration purposes."""
    args = parse_args()
    
    # Create a simple argument object that matches what run_benchmark expects
    class Args:
        pass
    
    benchmark_args = Args()
    benchmark_args.dataset = None  # Use built-in dataset
    benchmark_args.task = args.task
    benchmark_args.output_dir = args.output_dir
    benchmark_args.classical_models = ['random_forest']  # Only one classical model for speed
    benchmark_args.quantum_frameworks = ['qiskit']  # Only one quantum framework
    benchmark_args.quantum_model_types = ['variational']  # Only one quantum model type
    benchmark_args.hybrid_types = ['feature_hybrid']  # Only one hybrid approach
    benchmark_args.n_qubits = 4  # Small number of qubits for faster execution
    benchmark_args.save_models = False
    benchmark_args.verbose = True
    
    # Run the benchmark
    print("Running simplified demo benchmark...")
    print("This will train and evaluate a classical, quantum, and hybrid model")
    print("on a small built-in dataset.")
    print(f"Results will be saved to: {args.output_dir}")
    print("\nNOTE: This is a simplified demo. For a full comparison with more models,")
    print("use the benchmark.py script directly.\n")
    
    run_benchmark(benchmark_args)
    
    print("\nDemo complete! Check the results in the output directory.")
    print("To run a full benchmark with more models, use benchmark.py directly.")


if __name__ == "__main__":
    print("Quantum vs Classical Computing - Demo")
    print("====================================")
    run_demo() 