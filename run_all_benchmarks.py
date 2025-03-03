#!/usr/bin/env python3
"""
Quantum vs Classical Computing - Benchmark Runner

This script provides a unified interface to run all the different benchmark
tests in the project, including the ultimate benchmark dashboard.
"""

import os
import argparse
import subprocess
import sys
import webbrowser
from datetime import datetime

# Define all available benchmarks
BENCHMARKS = {
    "ultimate": {
        "script": "ultimate_benchmark_test.py",
        "description": "Comprehensive benchmark with complete interactive dashboard",
        "output_dir": "ultimate_benchmark_results",
        "dashboard": "benchmark_dashboard.html"
    },
    "enhanced": {
        "script": "enhanced_visualization_test.py",
        "description": "Enhanced visualizations with improved mock data",
        "output_dir": "visualization_results",
        "dashboard": None
    },
    "specific": {
        "script": "specific_visualization_test.py",
        "description": "Specific visualizations focused on time, resources, and practicality",
        "output_dir": "fixed_visualizations",
        "dashboard": "view_fixed.html"
    },
    "standard": {
        "script": "view_visualizations.py",
        "description": "Standard visualization viewer",
        "output_dir": None,
        "dashboard": "view_visualizations.html"
    }
}

def print_header():
    """Print the program header"""
    print("=" * 80)
    print(f"{'QUANTUM VS CLASSICAL COMPUTING BENCHMARK RUNNER':^80}")
    print(f"{'Compare performance across computing paradigms':^80}")
    print("=" * 80)
    print()

def list_benchmarks():
    """List all available benchmarks"""
    print("Available benchmarks:")
    print()
    for key, details in BENCHMARKS.items():
        print(f"  • {key:<12} - {details['description']}")
    print()

def run_benchmark(benchmark_name):
    """Run the specified benchmark"""
    if benchmark_name not in BENCHMARKS:
        print(f"Error: Benchmark '{benchmark_name}' not found.")
        list_benchmarks()
        return False
    
    benchmark = BENCHMARKS[benchmark_name]
    
    print(f"Running {benchmark_name} benchmark: {benchmark['description']}")
    print(f"Executing: {benchmark['script']}")
    print()
    
    # Run the script
    try:
        result = subprocess.run([sys.executable, benchmark['script']], 
                              check=True,
                              capture_output=False)
        
        print(f"\n✅ {benchmark_name} benchmark completed successfully.")
        
        # If there's a dashboard, offer to open it
        if benchmark['dashboard'] and benchmark['output_dir']:
            dashboard_path = os.path.join(benchmark['output_dir'], benchmark['dashboard'])
            if os.path.exists(dashboard_path):
                print(f"\nDashboard available at: {dashboard_path}")
                open_browser = input("Would you like to open it in your browser? (y/n): ").lower()
                if open_browser.startswith('y'):
                    dashboard_abs_path = os.path.abspath(dashboard_path)
                    webbrowser.open(f"file://{dashboard_abs_path}")
        
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Error running {benchmark_name} benchmark:")
        print(f"   Exit code: {e.returncode}")
        if e.stdout:
            print("\nOutput:")
            print(e.stdout.decode())
        if e.stderr:
            print("\nError:")
            print(e.stderr.decode())
        return False
    except Exception as e:
        print(f"\n❌ Unexpected error: {str(e)}")
        return False

def run_all_benchmarks():
    """Run all available benchmarks in sequence"""
    print("Running all benchmarks in sequence...\n")
    
    success_count = 0
    results = {}
    
    for name in BENCHMARKS:
        print(f"\n{'='*60}")
        print(f"Running {name} benchmark...")
        print(f"{'='*60}\n")
        
        start_time = datetime.now()
        success = run_benchmark(name)
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        results[name] = {
            "success": success,
            "duration": duration
        }
        
        if success:
            success_count += 1
    
    # Print summary
    print("\n\n")
    print("=" * 80)
    print(f"{'BENCHMARK SUMMARY':^80}")
    print("=" * 80)
    print(f"Completed {success_count} of {len(BENCHMARKS)} benchmarks.\n")
    
    for name, result in results.items():
        status = "✅ SUCCESS" if result["success"] else "❌ FAILED"
        print(f"{name:<12} - {status} ({result['duration']:.2f}s)")
    
    print("\nMost recommended benchmark is 'ultimate' for a complete analysis.")
    
    return success_count == len(BENCHMARKS)

def verify_dependencies():
    """Verify that all dependencies are installed"""
    required_modules = ['numpy', 'pandas', 'matplotlib']
    missing = []
    
    for module in required_modules:
        try:
            __import__(module)
        except ImportError:
            missing.append(module)
    
    if missing:
        print("❌ Missing required dependencies:")
        for module in missing:
            print(f"   - {module}")
        print("\nPlease install missing dependencies with:")
        print(f"pip install {' '.join(missing)}")
        return False
    
    return True

def main():
    """Main function"""
    print_header()
    
    # Create argument parser
    parser = argparse.ArgumentParser(
        description="Run Quantum vs Classical Computing benchmarks"
    )
    
    # Add arguments
    parser.add_argument(
        "benchmark", 
        nargs="?",
        choices=list(BENCHMARKS.keys()) + ["all"],
        default="ultimate",
        help="Benchmark to run (default: ultimate)"
    )
    parser.add_argument(
        "--list", 
        action="store_true",
        help="List available benchmarks"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # List benchmarks if requested
    if args.list:
        list_benchmarks()
        return 0
    
    # Verify dependencies
    if not verify_dependencies():
        return 1
    
    # Run the specified benchmark
    if args.benchmark == "all":
        success = run_all_benchmarks()
    else:
        success = run_benchmark(args.benchmark)
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 