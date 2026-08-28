"""Script to verify all required dependencies are installed before running Flask."""
required_packages = ["flask", "pandas", "numpy", "sklearn", "pickle"]

print("--- Checking Installed Libraries ---")
for pkg in required_packages:
    try:
        if pkg == "sklearn":
            import sklearn
            print(f"✅ scikit-learn version: {sklearn.__version__}")
        elif pkg == "pickle":
            import pickle
            print("✅ pickle module available")
        else:
            module = __import__(pkg)
            print(f"✅ {pkg} version: {module.__version__}")
    except ImportError as e:
        print(f"❌ Missing package: {pkg} -> Error: {e}")

print("-------------------------------------")