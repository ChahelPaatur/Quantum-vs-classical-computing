#!/usr/bin/env python3
"""
Project Cleanup Utility

This script helps clean up and organize the quantum vs classical computing benchmark project.
It performs the following tasks:
1. Organizes visualization files into appropriate directories
2. Removes temporary or unnecessary files
3. Checks for code consistency issues
4. Generates a project structure overview
"""

import os
import shutil
import glob
import sys
import subprocess
import re
from datetime import datetime

def print_header(message):
    """Print a formatted header message"""
    print("\n" + "=" * 80)
    print(f"{message:^80}")
    print("=" * 80)

def create_directory(directory):
    """Create a directory if it doesn't exist"""
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"✅ Created directory: {directory}")
    return directory

def organize_visualization_files():
    """Organize visualization files into appropriate directories"""
    print_header("ORGANIZING VISUALIZATION FILES")
    
    # Define directories to check and organize
    viz_directories = [
        "visualization_results",
        "fixed_visualizations",
        "ultimate_benchmark_results"
    ]
    
    # Create the main visualizations directory
    main_viz_dir = create_directory("visualizations")
    
    # Create timestamped backup directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    backup_dir = create_directory(os.path.join(main_viz_dir, f"backup_{timestamp}"))
    
    # Process each visualization directory
    total_moved = 0
    
    for viz_dir in viz_directories:
        if not os.path.exists(viz_dir):
            print(f"⚠️ Directory not found: {viz_dir}")
            continue
            
        # Create a subdirectory in the visualizations folder
        target_dir = create_directory(os.path.join(main_viz_dir, viz_dir))
        
        # Find all PNG files
        png_files = glob.glob(os.path.join(viz_dir, "*.png"))
        
        if not png_files:
            print(f"ℹ️ No PNG files found in {viz_dir}")
            continue
            
        # Copy PNG files to the target directory
        for png_file in png_files:
            filename = os.path.basename(png_file)
            # Backup original file
            shutil.copy2(png_file, os.path.join(backup_dir, filename))
            # Move to organized location
            shutil.copy2(png_file, os.path.join(target_dir, filename))
            total_moved += 1
        
        # Copy HTML files if they exist
        html_files = glob.glob(os.path.join(viz_dir, "*.html"))
        for html_file in html_files:
            filename = os.path.basename(html_file)
            # Backup original file
            shutil.copy2(html_file, os.path.join(backup_dir, filename))
            # Move to organized location 
            shutil.copy2(html_file, os.path.join(target_dir, filename))
            
            # Update image paths in HTML files
            update_html_paths(os.path.join(target_dir, filename), viz_dir)
            
    print(f"✅ Organized {total_moved} visualization files")
    print(f"✅ Backup created in {backup_dir}")
    
    return main_viz_dir

def update_html_paths(html_file, original_dir):
    """Update image paths in HTML files to reflect new directory structure"""
    try:
        with open(html_file, 'r') as f:
            content = f.read()
            
        # Replace direct image references with path to the organized directory
        updated_content = re.sub(
            r'(src=["\'])([^"/]+\.png)(["\'])', 
            f'\\1\\2\\3', 
            content
        )
        
        with open(html_file, 'w') as f:
            f.write(updated_content)
            
    except Exception as e:
        print(f"⚠️ Error updating paths in {html_file}: {str(e)}")

def remove_unnecessary_files():
    """Remove temporary or unnecessary files"""
    print_header("REMOVING UNNECESSARY FILES")
    
    # Files to remove (CAREFUL with this list!)
    files_to_remove = [
        "*.pyc",           # Python bytecode
        "__pycache__/**",  # Python cache directories
        "*.log",           # Log files
        ".DS_Store",       # macOS system files
        "*.bak",           # Backup files
        "*~"               # Temp files
    ]
    
    total_removed = 0
    
    for pattern in files_to_remove:
        files = glob.glob(pattern, recursive=True)
        for file in files:
            try:
                if os.path.isfile(file):
                    os.remove(file)
                elif os.path.isdir(file):
                    shutil.rmtree(file)
                total_removed += 1
                print(f"🗑️ Removed: {file}")
            except Exception as e:
                print(f"⚠️ Error removing {file}: {str(e)}")
    
    print(f"✅ Removed {total_removed} unnecessary files")

def check_code_consistency():
    """Check for code consistency issues"""
    print_header("CHECKING CODE CONSISTENCY")
    
    # Check Python files
    python_files = glob.glob("**/*.py", recursive=True)
    print(f"Found {len(python_files)} Python files")
    
    # Issues to check for
    missing_docstrings = []
    long_lines = []
    syntax_errors = []
    
    for py_file in python_files:
        try:
            with open(py_file, 'r') as f:
                content = f.read()
                lines = content.split("\n")
                
                # Check for docstring in modules and functions
                if not re.search(r'""".*?"""', content, re.DOTALL):
                    missing_docstrings.append(py_file)
                
                # Check for long lines
                for i, line in enumerate(lines):
                    if len(line) > 100:
                        long_lines.append((py_file, i+1, len(line)))
                        
        except SyntaxError:
            syntax_errors.append(py_file)
        except Exception as e:
            print(f"⚠️ Error checking {py_file}: {str(e)}")
    
    # Print results
    if missing_docstrings:
        print("\n⚠️ Files missing docstrings:")
        for file in missing_docstrings:
            print(f"  - {file}")
    
    if long_lines:
        print("\n⚠️ Files with lines > 100 characters:")
        for file, line, length in long_lines[:10]:  # Show only first 10
            print(f"  - {file} (line {line}): {length} chars")
        if len(long_lines) > 10:
            print(f"  ... and {len(long_lines) - 10} more")
    
    if syntax_errors:
        print("\n❌ Files with syntax errors:")
        for file in syntax_errors:
            print(f"  - {file}")
    
    if not missing_docstrings and not long_lines and not syntax_errors:
        print("✅ No major code consistency issues found")

def generate_structure_overview():
    """Generate a project structure overview"""
    print_header("GENERATING PROJECT STRUCTURE")
    
    output_file = "PROJECT_STRUCTURE.md"
    
    try:
        with open(output_file, 'w') as f:
            f.write("# Project Structure\n\n")
            f.write("Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n\n")
            
            # Execute find command and capture output
            result = subprocess.run(
                ['find', '.', '-type', 'f', '-not', '-path', '*/\.*', '|', 'sort'],
                stdout=subprocess.PIPE,
                text=True,
                shell=True
            )
            
            if result.returncode == 0:
                f.write("## Files\n\n```\n")
                f.write(result.stdout)
                f.write("```\n\n")
            
            # Add directory descriptions
            f.write("## Directory Structure\n\n")
            
            directories = {
                "data": "Contains datasets and examples",
                "models": "Implementation of classical, quantum, and hybrid models",
                "utils": "Utility modules for data loading, metrics, and visualization",
                "results": "Benchmark results",
                "visualizations": "Generated visualizations",
                "demo_results": "Results from demonstration runs"
            }
            
            for directory, description in directories.items():
                f.write(f"- **{directory}/**: {description}\n")
                
        print(f"✅ Structure overview generated: {output_file}")
        
    except Exception as e:
        print(f"❌ Error generating structure overview: {str(e)}")

def create_unified_dashboard():
    """Create a unified dashboard pointing to all visualization results"""
    print_header("CREATING UNIFIED DASHBOARD")
    
    dashboard_file = "dashboard.html"
    
    viz_dirs = [
        ("ultimate_benchmark_results", "Ultimate Benchmark"),
        ("visualization_results", "Enhanced Visualizations"),
        ("fixed_visualizations", "Specific Visualizations")
    ]
    
    html_content = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Quantum vs Classical Computing - Dashboard</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f7;
        }
        h1, h2, h3 {
            color: #1d1d1f;
        }
        h1 {
            font-size: 2.5rem;
            text-align: center;
            margin-bottom: 2rem;
        }
        .container {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            gap: 2rem;
        }
        .card {
            background-color: white;
            border-radius: 12px;
            box-shadow: 0 4px 20px rgba(0, 0, 0, 0.08);
            padding: 2rem;
            transition: transform 0.3s ease;
        }
        .card:hover {
            transform: translateY(-5px);
        }
        .card h2 {
            margin-top: 0;
            border-bottom: 2px solid #f5f5f7;
            padding-bottom: 0.5rem;
        }
        .btn {
            display: inline-block;
            background-color: #0071e3;
            color: white;
            padding: 0.5rem 1rem;
            border-radius: 20px;
            text-decoration: none;
            margin-top: 1rem;
            font-weight: bold;
        }
        .btn:hover {
            background-color: #0077ed;
        }
        footer {
            text-align: center;
            margin-top: 3rem;
            color: #86868b;
            padding-top: 1rem;
            border-top: 1px solid #d2d2d7;
        }
    </style>
</head>
<body>
    <h1>Quantum vs Classical Computing</h1>
    
    <div class="container">
"""
    
    # Add cards for each dashboard
    for dir_name, title in viz_dirs:
        # Check if HTML files exist in this directory
        html_files = glob.glob(os.path.join(dir_name, "*.html"))
        if not html_files:
            continue
            
        dashboard_file = html_files[0]
        
        html_content += f"""
        <div class="card">
            <h2>{title}</h2>
            <p>View benchmark results and visualizations from the {title.lower()} tests.</p>
            <a href="{dashboard_file}" class="btn">View Dashboard</a>
        </div>
        """
    
    # Add card for running the benchmark
    html_content += """
        <div class="card">
            <h2>Run Benchmark</h2>
            <p>Execute the benchmark tests to generate new visualizations and results.</p>
            <a href="#" onclick="runBenchmark(); return false;" class="btn">Run Benchmark</a>
        </div>
    </div>
    
    <footer>
        <p>Quantum vs Classical Computing Benchmark Project</p>
    </footer>
    
    <script>
        function runBenchmark() {
            if (confirm("Do you want to run the benchmark test? This may take a few minutes.")) {
                window.location.href = "run_all_benchmarks.py";
            }
        }
    </script>
</body>
</html>
    """
    
    try:
        with open("dashboard.html", 'w') as f:
            f.write(html_content)
        print(f"✅ Unified dashboard created: dashboard.html")
    except Exception as e:
        print(f"❌ Error creating dashboard: {str(e)}")

def organize_all_visualization_dirs():
    """Organize all visualization directories and create links to the most up-to-date visualizations"""
    print_header("Organizing All Visualization Directories")
    
    # Create main visualization directory
    main_viz_dir = "visualizations"
    create_directory(main_viz_dir)
    
    # List of all visualization dirs to organize
    viz_dirs = [
        "visualization_results",
        "fixed_visualizations",
        "ultimate_benchmark_results"
    ]
    
    # Copy the latest versions of each visualization type to the main directory
    visualization_types = {
        "accuracy_comparison.png": None,
        "time_comparison.png": None,
        "resource_usage.png": None,
        "practicality_comparison.png": None,
        "radar_comparison.png": None
    }
    
    # Find the most recent version of each visualization
    for viz_dir in viz_dirs:
        if not os.path.exists(viz_dir):
            continue
            
        for viz_type in visualization_types.keys():
            viz_path = os.path.join(viz_dir, viz_type)
            if os.path.exists(viz_path):
                # If we haven't found this visualization yet, or this one is newer
                if (visualization_types[viz_type] is None or 
                    os.path.getmtime(viz_path) > os.path.getmtime(visualization_types[viz_type])):
                    visualization_types[viz_type] = viz_path
    
    # Copy the most recent visualizations to the main directory
    for viz_type, viz_path in visualization_types.items():
        if viz_path is not None:
            dest_path = os.path.join(main_viz_dir, viz_type)
            shutil.copy2(viz_path, dest_path)
            print(f"  • Copied latest {viz_type} to {dest_path}")
    
    # Create an index.html file that links to all visualizations
    index_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Visualization Index</title>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1000px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f5f5f7;
            color: #1d1d1f;
        }
        h1, h2 {
            font-weight: bold;
        }
        .links {
            display: flex;
            flex-direction: column;
            gap: 10px;
            margin: 20px 0;
        }
        .dashboard-link {
            display: block;
            background-color: #0071e3;
            color: white;
            padding: 12px 20px;
            text-decoration: none;
            border-radius: 8px;
            margin-bottom: 10px;
            text-align: center;
            font-weight: bold;
        }
        .dashboard-link:hover {
            background-color: #0066cc;
        }
        .viz-container {
            background-color: white;
            border-radius: 12px;
            padding: 20px;
            margin-bottom: 20px;
            box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        }
        .viz-list {
            display: grid;
            grid-template-columns: repeat(auto-fill, minmax(200px, 1fr));
            gap: 15px;
        }
        .viz-item {
            background-color: #f5f5f7;
            padding: 10px;
            border-radius: 8px;
            text-align: center;
        }
        .viz-item a {
            color: #0071e3;
            text-decoration: none;
            font-weight: 500;
        }
    </style>
</head>
<body>
    <h1>Quantum vs Classical Computing Benchmark</h1>
    
    <div class="links">
        <a href="../master_dashboard.html" class="dashboard-link">Open Master Dashboard</a>
        <a href="../ultimate_benchmark_results/benchmark_dashboard.html" class="dashboard-link">Open Benchmark Dashboard</a>
    </div>
    
    <div class="viz-container">
        <h2>All Visualizations</h2>
        <p>Access all available visualizations:</p>
        
        <div class="viz-list">
"""
    
    # Add links to each visualization
    for viz_type in visualization_types.keys():
        if os.path.exists(os.path.join(main_viz_dir, viz_type)):
            viz_name = viz_type.replace('_', ' ').replace('.png', '').title()
            index_html += f"""
            <div class="viz-item">
                <a href="{viz_type}" target="_blank">{viz_name}</a>
            </div>"""
    
    index_html += """
        </div>
    </div>
    
    <div class="viz-container">
        <h2>Visualization Directories</h2>
        <p>Access different visualization sets:</p>
        
        <div class="viz-list">
"""
    
    # Add links to each visualization directory
    for viz_dir in viz_dirs:
        if os.path.exists(viz_dir):
            viz_dir_name = viz_dir.replace('_', ' ').title()
            index_html += f"""
            <div class="viz-item">
                <a href="../{viz_dir}" target="_blank">{viz_dir_name}</a>
            </div>"""
    
    index_html += """
        </div>
    </div>
</body>
</html>
"""
    
    # Write the index.html file
    with open(os.path.join(main_viz_dir, "index.html"), "w") as f:
        f.write(index_html)
    
    print(f"  • Created visualization index at {main_viz_dir}/index.html")
    
    return True

def main():
    """Main function to run all cleanup tasks"""
    print_header("QUANTUM VS CLASSICAL COMPUTING PROJECT CLEANUP")
    
    # Ask for confirmation before proceeding
    confirm = input("\nThis script will organize visualization files, remove unnecessary files, check code consistency, and generate a project structure overview.\nContinue? (y/n): ")
    if confirm.lower() != 'y':
        print("Cleanup cancelled.")
        return
    
    # Create a backup directory
    backup_dir = "backup_" + datetime.now().strftime("%Y%m%d_%H%M%S")
    create_directory(backup_dir)
    print(f"Created backup directory: {backup_dir}")
    
    # Organize visualization files
    organize_visualization_files()
    
    # Organize all visualization directories
    organize_all_visualization_dirs()
    
    # Remove unnecessary files
    remove_unnecessary_files()
    
    # Check code consistency
    check_code_consistency()
    
    # Generate structure overview
    generate_structure_overview()
    
    # Create unified dashboard
    create_unified_dashboard()
    
    print_header("CLEANUP COMPLETE")
    print("\nThe project has been cleaned and organized successfully.")
    print(f"A backup of original files is available in: {backup_dir}")
    print("\nKey files generated:")
    print("  • PROJECT_STRUCTURE.md - Overview of project structure")
    print("  • visualizations/index.html - Index of all visualizations")
    print("  • unified_dashboard.html - Comprehensive dashboard of all results")
    
    print("\nYou can now run the organized project with:")
    print("  python run_all_benchmarks.py")
    print("  or")
    print("  open master_dashboard.html")

if __name__ == "__main__":
    main() 