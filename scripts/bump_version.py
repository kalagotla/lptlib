#!/usr/bin/env python3
"""
Version bumping script for lptlib
Usage: python scripts/bump_version.py [major|minor|patch|alpha|beta|rc]
"""

import re
import sys
import os
from pathlib import Path

def get_current_version():
    """Get current version from pyproject.toml"""
    pyproject_path = Path("pyproject.toml")
    if not pyproject_path.exists():
        raise FileNotFoundError("pyproject.toml not found")
    
    with open(pyproject_path, 'r') as f:
        content = f.read()
    
    match = re.search(r'version = "([^"]+)"', content)
    if not match:
        raise ValueError("Could not find version in pyproject.toml")
    
    return match.group(1)

def bump_version(current_version, bump_type):
    """Bump version based on type"""
    # Parse version (e.g., "0.0.5a4" -> (0, 0, 5, 'a', 4))
    pattern = r'(\d+)\.(\d+)\.(\d+)([a-z]+)?(\d+)?'
    match = re.match(pattern, current_version)
    
    if not match:
        raise ValueError(f"Invalid version format: {current_version}")
    
    major, minor, patch, prerelease_type, prerelease_num = match.groups()
    major, minor, patch = int(major), int(minor), int(patch)
    
    if prerelease_num:
        prerelease_num = int(prerelease_num)
    
    if bump_type == "major":
        major += 1
        minor = 0
        patch = 0
        prerelease_type = None
        prerelease_num = None
    elif bump_type == "minor":
        minor += 1
        patch = 0
        prerelease_type = None
        prerelease_num = None
    elif bump_type == "patch":
        patch += 1
        prerelease_type = None
        prerelease_num = None
    elif bump_type in ["alpha", "beta", "rc"]:
        # Convert full names to short forms for comparison
        current_prerelease_type = prerelease_type
        if current_prerelease_type == "alpha":
            current_prerelease_type = "a"
        elif current_prerelease_type == "beta":
            current_prerelease_type = "b"
        
        target_type = bump_type
        if target_type == "alpha":
            target_type = "a"
        elif target_type == "beta":
            target_type = "b"
        
        if current_prerelease_type == target_type:
            # Same prerelease type, increment number
            prerelease_num = (prerelease_num or 0) + 1
        else:
            # Different prerelease type, start from 0
            prerelease_num = 0
        
        prerelease_type = target_type
    else:
        raise ValueError(f"Invalid bump type: {bump_type}")
    
    # Build new version string
    new_version = f"{major}.{minor}.{patch}"
    if prerelease_type and prerelease_num is not None:
        new_version += f"{prerelease_type}{prerelease_num}"
    
    return new_version

def update_version_files(new_version):
    """Update version in pyproject.toml and setup.py"""
    # Update pyproject.toml
    pyproject_path = Path("pyproject.toml")
    with open(pyproject_path, 'r') as f:
        content = f.read()
    
    content = re.sub(r'version = "[^"]+"', f'version = "{new_version}"', content)
    
    with open(pyproject_path, 'w') as f:
        f.write(content)
    
    # Update setup.py
    setup_path = Path("setup.py")
    if setup_path.exists():
        with open(setup_path, 'r') as f:
            content = f.read()
        
        content = re.sub(r'version="[^"]+"', f'version="{new_version}"', content)
        
        with open(setup_path, 'w') as f:
            f.write(content)
    
    print(f"Updated version to {new_version} in pyproject.toml and setup.py")

def main():
    if len(sys.argv) != 2:
        print("Usage: python scripts/bump_version.py [major|minor|patch|alpha|beta|rc]")
        sys.exit(1)
    
    bump_type = sys.argv[1].lower()
    valid_types = ["major", "minor", "patch", "alpha", "beta", "rc"]
    
    if bump_type not in valid_types:
        print(f"Invalid bump type. Must be one of: {', '.join(valid_types)}")
        sys.exit(1)
    
    try:
        current_version = get_current_version()
        new_version = bump_version(current_version, bump_type)
        update_version_files(new_version)
        print(f"Version bumped from {current_version} to {new_version}")
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
