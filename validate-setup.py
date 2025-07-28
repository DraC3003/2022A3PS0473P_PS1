#!/usr/bin/env python3
"""
Validation script to check VSCode setup configuration
"""

import os
import sys
import json
import subprocess

def check_python_version():
    """Check if Python version is suitable"""
    version = sys.version_info
    if version.major >= 3 and version.minor >= 8:
        print(f"✅ Python {version.major}.{version.minor}.{version.micro} - OK")
        return True
    else:
        print(f"❌ Python {version.major}.{version.minor}.{version.micro} - Need Python 3.8+")
        return False

def check_file_exists(filepath):
    """Check if a file exists"""
    if os.path.exists(filepath):
        print(f"✅ {filepath} - EXISTS")
        return True
    else:
        print(f"❌ {filepath} - MISSING")
        return False

def validate_json_file(filepath):
    """Validate JSON file syntax"""
    try:
        with open(filepath, 'r') as f:
            json.load(f)
        print(f"✅ {filepath} - VALID JSON")
        return True
    except Exception as e:
        print(f"❌ {filepath} - INVALID JSON: {e}")
        return False

def check_directories():
    """Check if required directories exist"""
    dirs = [
        "CONT",
        "AABIR_SARKAR_PS-I/Risk prediction app and analysis on Fetal health and Diabetes",
        ".vscode"
    ]
    
    all_exist = True
    for dir_path in dirs:
        if os.path.exists(dir_path):
            print(f"✅ Directory {dir_path} - EXISTS")
        else:
            print(f"❌ Directory {dir_path} - MISSING")
            all_exist = False
    
    return all_exist

def check_vscode_config():
    """Check VSCode configuration files"""
    vscode_files = [
        ".vscode/settings.json",
        ".vscode/launch.json",
        ".vscode/tasks.json"
    ]
    
    all_valid = True
    for file_path in vscode_files:
        if check_file_exists(file_path):
            if not validate_json_file(file_path):
                all_valid = False
        else:
            all_valid = False
    
    return all_valid

def check_requirements_files():
    """Check if requirements.txt files exist"""
    req_files = [
        "CONT/requirements.txt",
        "AABIR_SARKAR_PS-I/Risk prediction app and analysis on Fetal health and Diabetes/requirements.txt"
    ]
    
    all_exist = True
    for req_file in req_files:
        if not check_file_exists(req_file):
            all_exist = False
    
    return all_exist

def check_main_files():
    """Check if main application files exist"""
    main_files = [
        "CONT/run.py",
        "AABIR_SARKAR_PS-I/Risk prediction app and analysis on Fetal health and Diabetes/app3.py"
    ]
    
    all_exist = True
    for main_file in main_files:
        if not check_file_exists(main_file):
            all_exist = False
    
    return all_exist

def check_setup_scripts():
    """Check if setup scripts exist and are executable"""
    scripts = [
        "setup-vscode.sh",
        "setup-vscode.bat"
    ]
    
    all_good = True
    for script in scripts:
        if check_file_exists(script):
            if script.endswith('.sh'):
                # Check if executable on Unix systems
                if os.name != 'nt' and not os.access(script, os.X_OK):
                    print(f"⚠️  {script} - Not executable (run: chmod +x {script})")
                else:
                    print(f"✅ {script} - Executable")
        else:
            all_good = False
    
    return all_good

def main():
    """Main validation function"""
    print("🔍 Validating VSCode Setup Configuration")
    print("=" * 50)
    
    checks = [
        ("Python Version", check_python_version),
        ("Directory Structure", check_directories),
        ("VSCode Configuration", check_vscode_config),
        ("Requirements Files", check_requirements_files),
        ("Main Application Files", check_main_files),
        ("Setup Scripts", check_setup_scripts)
    ]
    
    results = []
    for check_name, check_func in checks:
        print(f"\n📋 Checking {check_name}:")
        result = check_func()
        results.append((check_name, result))
    
    print("\n" + "=" * 50)
    print("📊 VALIDATION SUMMARY")
    print("=" * 50)
    
    all_passed = True
    for check_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status} - {check_name}")
        if not result:
            all_passed = False
    
    print("\n" + "=" * 50)
    if all_passed:
        print("🎉 ALL CHECKS PASSED! Your VSCode setup is ready.")
        print("\n📚 Next steps:")
        print("1. Run setup script: ./setup-vscode.sh (Linux/Mac) or setup-vscode.bat (Windows)")
        print("2. Open in VSCode: code .")
        print("3. Install recommended extensions when prompted")
        print("4. See VSCODE_SETUP_GUIDE.md for detailed instructions")
    else:
        print("⚠️  SOME CHECKS FAILED. Please review the issues above.")
        print("\n📖 For help, see TROUBLESHOOTING.md")
    
    return 0 if all_passed else 1

if __name__ == "__main__":
    sys.exit(main())