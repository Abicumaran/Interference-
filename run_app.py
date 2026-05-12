import os, sys, subprocess

def main():
    here = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(here, "app.py")
    subprocess.Popen([sys.executable, "-m", "streamlit", "run", app_path, "--server.headless=true"])

if __name__ == "__main__":
    main()
