#!/usr/bin/env python3
"""
Complete release script for lptlib
Usage: python scripts/release.py [major|minor|patch|alpha|beta|rc] [--test|--prod]
"""

import subprocess
import sys
import os
import shutil
from pathlib import Path

def run_command(cmd, check=True):
    """Run a command and return the result"""
    print(f"Running: {cmd}")
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if check and result.returncode != 0:
        print(f"Error running command: {cmd}")
        print(f"stdout: {result.stdout}")
        print(f"stderr: {result.stderr}")
        sys.exit(1)
    return result

def clean_build():
    """Clean previous builds"""
    print("Cleaning previous builds...")
    dirs_to_clean = ["dist", "build", "lptlib.egg-info"]
    for dir_name in dirs_to_clean:
        if os.path.exists(dir_name):
            shutil.rmtree(dir_name)
            print(f"Removed {dir_name}/")
    
    # Clean egg-info directories
    for egg_info in Path(".").glob("**/*.egg-info"):
        shutil.rmtree(egg_info)
        print(f"Removed {egg_info}/")

def bump_version(bump_type):
    """Bump version using the bump_version script"""
    print(f"Bumping version ({bump_type})...")
    run_command(f"python scripts/bump_version.py {bump_type}")

def build_package():
    """Build the package"""
    print("Building package...")
    run_command("python setup.py sdist bdist_wheel")

def check_package():
    """Check the package"""
    print("Checking package...")
    run_command("twine check dist/*")

def upload_to_testpypi():
    """Upload to TestPyPI"""
    print("Uploading to TestPyPI...")
    run_command("twine upload --repository testpypi dist/*")

def upload_to_pypi():
    """Upload to PyPI"""
    print("Uploading to PyPI...")
    run_command("twine upload dist/*")

def test_installation():
    """Test installation from TestPyPI"""
    print("Testing installation from TestPyPI...")
    run_command("pip install --index-url https://test.pypi.org/simple/ lptlib --force-reinstall")

def create_git_tag():
    """Create a git tag for the release"""
    # Get version from pyproject.toml
    with open("pyproject.toml", 'r') as f:
        content = f.read()
    
    import re
    match = re.search(r'version = "([^"]+)"', content)
    if match:
        version = match.group(1)
        tag_name = f"v{version}"
        print(f"Creating git tag: {tag_name}")
        run_command(f"git add .")
        run_command(f"git commit -m 'Release {version}'")
        run_command(f"git tag {tag_name}")
        print(f"Created tag: {tag_name}")
        print(f"Push with: git push origin main --tags")

def main():
    if len(sys.argv) < 2:
        print("Usage: python scripts/release.py [major|minor|patch|alpha|beta|rc] [--test|--prod]")
        print("  --test: Upload to TestPyPI only")
        print("  --prod: Upload to production PyPI")
        sys.exit(1)
    
    bump_type = sys.argv[1]
    upload_target = "test"  # Default to test
    
    if len(sys.argv) > 2:
        if sys.argv[2] == "--prod":
            upload_target = "prod"
        elif sys.argv[2] == "--test":
            upload_target = "test"
        else:
            print("Invalid upload target. Use --test or --prod")
            sys.exit(1)
    
    print(f"Starting release process:")
    print(f"  Bump type: {bump_type}")
    print(f"  Upload target: {upload_target}")
    print()
    
    try:
        # Step 1: Clean
        clean_build()
        
        # Step 2: Bump version
        bump_version(bump_type)
        
        # Step 3: Build
        build_package()
        
        # Step 4: Check
        check_package()
        
        # Step 5: Upload
        if upload_target == "test":
            upload_to_testpypi()
            print("\n✅ Successfully uploaded to TestPyPI!")
            print("Test installation with:")
            print("  pip install --index-url https://test.pypi.org/simple/ lptlib")
        else:
            upload_to_pypi()
            print("\n✅ Successfully uploaded to PyPI!")
            print("Install with:")
            print("  pip install lptlib")
        
        # Step 6: Create git tag
        create_git_tag()
        
        print(f"\n🎉 Release completed successfully!")
        
    except KeyboardInterrupt:
        print("\n❌ Release cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Release failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
