#!/usr/bin/env python3
"""
Test Auto-Fill Functionality

This script demonstrates the auto-fill functionality by running it on the
paper_with_topics.json file to show what information can be extracted.
"""

import subprocess
import sys

def main():
    print("🚀 Testing Auto-Fill Paper Details Script")
    print("=" * 50)
    
    # First, show what would be updated (dry run)
    print("🔍 Running dry-run to show what can be updated...")
    try:
        result = subprocess.run([
            sys.executable, "auto_fill_paper_details.py", 
            "--dry-run", 
            "--verbose"
        ], 
        capture_output=True, 
        text=True, 
        cwd="."
        )
        
        print("STDOUT:")
        print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode != 0:
            print(f"❌ Script failed with return code: {result.returncode}")
        else:
            print("✅ Dry run completed successfully!")
            
        # Ask if user wants to run the actual update
        print("\n" + "=" * 50)
        response = input("Would you like to run the actual update? (y/n): ")
        
        if response.lower().startswith('y'):
            print("\n🔄 Running actual update with backup...")
            result = subprocess.run([
                sys.executable, "auto_fill_paper_details.py", 
                "--backup", 
                "--verbose"
            ], 
            capture_output=True, 
            text=True, 
            cwd="."
            )
            
            print("STDOUT:")
            print(result.stdout)
            
            if result.stderr:
                print("STDERR:")
                print(result.stderr)
            
            if result.returncode != 0:
                print(f"❌ Update failed with return code: {result.returncode}")
            else:
                print("✅ Update completed successfully!")
        else:
            print("👍 No changes made to your data.")
            
    except FileNotFoundError:
        print("❌ Error: auto_fill_paper_details.py not found!")
        print("Make sure the script is in the current directory.")
    except Exception as e:
        print(f"❌ Error running script: {e}")

if __name__ == "__main__":
    main()