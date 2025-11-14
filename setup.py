#!/usr/bin/env python3
"""
Setup script for WhatsApp Django Analytics Project
This script helps set up the project environment and dependencies.
"""

import os
import sys
import subprocess
import platform

def run_command(command, description):
    """Run a command and handle errors"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed: {e}")
        print(f"Error output: {e.stderr}")
        return False

def check_python_version():
    """Check if Python version is compatible"""
    print("🐍 Checking Python version...")
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("❌ Python 3.8 or higher is required")
        print(f"Current version: {version.major}.{version.minor}.{version.micro}")
        return False
    print(f"✅ Python {version.major}.{version.minor}.{version.micro} is compatible")
    return True

def create_virtual_environment():
    """Create virtual environment"""
    if not os.path.exists('venv'):
        return run_command('python -m venv venv', 'Creating virtual environment')
    else:
        print("✅ Virtual environment already exists")
        return True

def activate_venv_and_install():
    """Activate virtual environment and install dependencies"""
    if platform.system() == "Windows":
        activate_cmd = "venv\\Scripts\\activate"
        pip_cmd = "venv\\Scripts\\pip"
    else:
        activate_cmd = "source venv/bin/activate"
        pip_cmd = "venv/bin/pip"
    
    # Install dependencies
    if not run_command(f"{pip_cmd} install --upgrade pip", "Upgrading pip"):
        return False
    
    if not run_command(f"{pip_cmd} install -r requirements.txt", "Installing project dependencies"):
        return False
    
    return True

def run_django_setup():
    """Run Django setup commands"""
    if platform.system() == "Windows":
        python_cmd = "venv\\Scripts\\python"
    else:
        python_cmd = "venv/bin/python"
    
    # Run migrations
    if not run_command(f"{python_cmd} manage.py makemigrations", "Creating Django migrations"):
        return False
    
    if not run_command(f"{python_cmd} manage.py migrate", "Running Django migrations"):
        return False
    
    return True

def create_env_file():
    """Create .env file with default settings"""
    env_content = """# Django Settings
SECRET_KEY=django-insecure-7xtf^8mx%38rgf_xv&*+oqfk1zf746akl@+*vli@w2w_my49az
DEBUG=True
ALLOWED_HOSTS=127.0.0.1,localhost
CSRF_TRUSTED_ORIGINS=http://127.0.0.1:8000,http://localhost:8000

# Google Gemini API (Optional - for AI features)
# GEMINI_API_KEY=your-gemini-api-key-here
"""
    
    if not os.path.exists('.env'):
        try:
            with open('.env', 'w') as f:
                f.write(env_content)
            print("✅ Created .env file with default settings")
            return True
        except Exception as e:
            print(f"❌ Failed to create .env file: {e}")
            return False
    else:
        print("✅ .env file already exists")
        return True

def main():
    """Main setup function"""
    print("🚀 Setting up WhatsApp Django Analytics Project")
    print("=" * 50)
    
    # Check Python version
    if not check_python_version():
        sys.exit(1)
    
    # Create virtual environment
    if not create_virtual_environment():
        sys.exit(1)
    
    # Install dependencies
    if not activate_venv_and_install():
        sys.exit(1)
    
    # Create .env file
    if not create_env_file():
        sys.exit(1)
    
    # Run Django setup
    if not run_django_setup():
        sys.exit(1)
    
    print("\n" + "=" * 50)
    print("🎉 Setup completed successfully!")
    print("\n📋 Next steps:")
    print("1. Activate the virtual environment:")
    if platform.system() == "Windows":
        print("   venv\\Scripts\\activate")
    else:
        print("   source venv/bin/activate")
    print("2. Start the development server:")
    print("   python manage.py runserver")
    print("3. Open your browser and go to: http://127.0.0.1:8000")
    print("\n💡 Optional: Add your GEMINI_API_KEY to .env file for AI features")
    print("=" * 50)

if __name__ == "__main__":
    main()

