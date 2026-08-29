#!/usr/bin/env python3
"""
Windows Executable Build Script for Apple Detection GUI
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path

def main():
    # 1. Ensure PyInstaller is installed
    try:
        import PyInstaller
        print(f"[OK] PyInstaller is already installed (version {PyInstaller.__version__})")
    except ImportError:
        print("[INFO] PyInstaller not found. Installing now...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
            print("[OK] PyInstaller installed successfully!")
        except Exception as e:
            print(f"[ERROR] Failed to install PyInstaller: {e}")
            sys.exit(1)

    import PyInstaller.__main__

    # 2. Define build parameters
    script_path = "gui_app.py"
    app_name = "AppleDetectionApp"
    
    # Check mode (defaults to --onedir for PyTorch package speed and stability)
    onefile_mode = "--onefile" in sys.argv or "-f" in sys.argv
    mode_str = "--onefile" if onefile_mode else "--onedir"
    
    print(f"\n[BUILD] Packaging '{script_path}' into a Windows Executable...")
    print(f"Mode: {mode_str}")
    print("Please wait, bundling deep learning libraries (PyTorch/OpenCV/YOLO) can take several minutes...\n")

    # Set up PyInstaller command-line arguments
    args = [
        script_path,
        f"--name={app_name}",
        mode_str,
        "--noconsole",          # Hide console window
        "--noconfirm",          # Overwrite output directory without confirmation
        "--clean",              # Clean PyInstaller cache
        "--collect-data=ultralytics", # Collect YOLO configurations, assets, and defaults
        "--collect-data=torch",       # Collect PyTorch resources
    ]

    # Run PyInstaller compiler
    try:
        PyInstaller.__main__.run(args)
        print("\n[OK] PyInstaller compilation process finished successfully!")
    except Exception as e:
        print(f"\n[ERROR] PyInstaller compilation failed: {e}")
        sys.exit(1)

    # 3. Resource deployment
    # We want to copy the local 'weights' and 'apple_dataset' folders into the distribution directory
    # so that the packaged executable can load the YOLO model and access test files seamlessly.
    dist_base = Path("dist")
    dest_dir = dist_base / app_name if not onefile_mode else dist_base
    
    print(f"\n[POST-BUILD] Deploying resources to {dest_dir.resolve()}...")
    
    # Copy weights directory
    src_weights = Path("weights")
    dst_weights = dest_dir / "weights"
    if src_weights.exists():
        try:
            if dst_weights.exists():
                shutil.rmtree(dst_weights)
            shutil.copytree(src_weights, dst_weights)
            print("[COPY] Copied 'weights' folder (with best.pt) to output directory.")
        except Exception as e:
            print(f"[WARN] Failed to copy 'weights' directory: {e}")
    else:
        print("[WARN] Local 'weights' directory not found.")
        
    # Copy apple_dataset directory if present
    src_dataset = Path("apple_dataset")
    dst_dataset = dest_dir / "apple_dataset"
    if src_dataset.exists():
        try:
            if dst_dataset.exists():
                shutil.rmtree(dst_dataset)
            shutil.copytree(src_dataset, dst_dataset)
            print("[COPY] Copied 'apple_dataset' folder to output directory.")
        except Exception as e:
            print(f"[WARN] Failed to copy 'apple_dataset' directory: {e}")
    else:
        print("[INFO] Note: 'apple_dataset' folder not found.")

    print("\n" + "="*60)
    print("EXECUTE COMPILED APP")
    print("="*60)
    exe_file = dest_dir / f"{app_name}.exe"
    print(f"Executable Location: {exe_file.resolve()}")
    print("\nDouble-click the executable to launch your Windows Form application.")
    print("="*60 + "\n")

if __name__ == "__main__":
    main()
