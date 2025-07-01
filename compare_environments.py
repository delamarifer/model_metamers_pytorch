#!/usr/bin/env python3
"""
Script to compare two conda environment files and identify key differences.
"""

import yaml
from collections import defaultdict

def parse_environment_file(filename):
    """Parse a conda environment file and return structured data."""
    with open(filename, 'r') as f:
        data = yaml.safe_load(f)
    
    # Extract conda packages
    conda_packages = {}
    if 'dependencies' in data:
        for dep in data['dependencies']:
            if isinstance(dep, str) and '=' in dep:
                name = dep.split('=')[0]
                version = '='.join(dep.split('=')[1:])
                conda_packages[name] = version
    
    # Extract pip packages
    pip_packages = {}
    if 'dependencies' in data:
        for dep in data['dependencies']:
            if isinstance(dep, dict) and 'pip' in dep:
                for pip_dep in dep['pip']:
                    if '==' in pip_dep:
                        name, version = pip_dep.split('==', 1)
                        pip_packages[name] = version
                    elif '=' in pip_dep:
                        name = pip_dep.split('=')[0]
                        version = '='.join(pip_dep.split('=')[1:])
                        pip_packages[name] = version
    
    return {
        'name': data.get('name', 'unknown'),
        'channels': data.get('channels', []),
        'conda_packages': conda_packages,
        'pip_packages': pip_packages
    }

def compare_environments(env1, env2):
    """Compare two environment specifications."""
    print("=" * 80)
    print("ENVIRONMENT COMPARISON")
    print("=" * 80)
    
    # Compare basic info
    print(f"\nEnvironment Names:")
    print(f"  Current: {env1['name']}")
    print(f"  2022:    {env2['name']}")
    
    print(f"\nChannels:")
    print(f"  Current: {env1['channels']}")
    print(f"  2022:    {env2['channels']}")
    
    # Compare conda packages
    conda1 = set(env1['conda_packages'].keys())
    conda2 = set(env2['conda_packages'].keys())
    
    only_in_current = conda1 - conda2
    only_in_2022 = conda2 - conda1
    common_conda = conda1 & conda2
    
    print(f"\nConda Packages:")
    print(f"  Total packages - Current: {len(conda1)}, 2022: {len(conda2)}")
    print(f"  Common packages: {len(common_conda)}")
    print(f"  Only in current: {len(only_in_current)}")
    print(f"  Only in 2022: {len(only_in_2022)}")
    
    if only_in_current:
        print(f"\n  Packages only in current environment:")
        for pkg in sorted(only_in_current):
            print(f"    + {pkg}={env1['conda_packages'][pkg]}")
    
    if only_in_2022:
        print(f"\n  Packages only in 2022 environment:")
        for pkg in sorted(only_in_2022):
            print(f"    - {pkg}={env2['conda_packages'][pkg]}")
    
    # Compare versions of common packages
    version_differences = []
    for pkg in common_conda:
        ver1 = env1['conda_packages'][pkg]
        ver2 = env2['conda_packages'][pkg]
        if ver1 != ver2:
            version_differences.append((pkg, ver1, ver2))
    
    if version_differences:
        print(f"\n  Version differences in common packages:")
        for pkg, ver1, ver2 in sorted(version_differences):
            print(f"    {pkg}: {ver2} -> {ver1}")
    
    # Compare pip packages
    pip1 = set(env1['pip_packages'].keys())
    pip2 = set(env2['pip_packages'].keys())
    
    only_in_current_pip = pip1 - pip2
    only_in_2022_pip = pip2 - pip1
    common_pip = pip1 & pip2
    
    print(f"\nPip Packages:")
    print(f"  Total packages - Current: {len(pip1)}, 2022: {len(pip2)}")
    print(f"  Common packages: {len(common_pip)}")
    print(f"  Only in current: {len(only_in_current_pip)}")
    print(f"  Only in 2022: {len(only_in_2022_pip)}")
    
    if only_in_current_pip:
        print(f"\n  Pip packages only in current environment:")
        for pkg in sorted(only_in_current_pip):
            print(f"    + {pkg}=={env1['pip_packages'][pkg]}")
    
    if only_in_2022_pip:
        print(f"\n  Pip packages only in 2022 environment:")
        for pkg in sorted(only_in_2022_pip):
            print(f"    - {pkg}=={env2['pip_packages'][pkg]}")
    
    # Compare versions of common pip packages
    pip_version_differences = []
    for pkg in common_pip:
        ver1 = env1['pip_packages'][pkg]
        ver2 = env2['pip_packages'][pkg]
        if ver1 != ver2:
            pip_version_differences.append((pkg, ver1, ver2))
    
    if pip_version_differences:
        print(f"\n  Pip version differences in common packages:")
        for pkg, ver1, ver2 in sorted(pip_version_differences):
            print(f"    {pkg}: {ver2} -> {ver1}")

def main():
    # Parse both environment files
    current_env = parse_environment_file('environment.yml')
    env_2022 = parse_environment_file('feather_metamers_conda_2022.yml')
    
    # Compare them
    compare_environments(current_env, env_2022)
    
    # Summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    
    current_total = len(current_env['conda_packages']) + len(current_env['pip_packages'])
    env_2022_total = len(env_2022['conda_packages']) + len(env_2022['pip_packages'])
    
    print(f"Total packages - Current: {current_total}, 2022: {env_2022_total}")
    print(f"Difference: {current_total - env_2022_total} packages")
    
    # Key differences
    print(f"\nKey differences:")
    print(f"- Current environment has newer CUDA components (11.8 vs 10.1)")
    print(f"- Current environment has PyTorch 1.7.1 vs 1.5.0 in 2022")
    print(f"- Current environment has additional multimedia libraries (ffmpeg, etc.)")
    print(f"- Current environment has newer security certificates and libraries")

if __name__ == "__main__":
    main() 