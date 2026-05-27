import importlib
import socket
import sys

try:
    # packaging is a dependency of pip; typically available
    from packaging.requirements import Requirement
    from packaging.version import Version
except Exception:
    print("❌ 'packaging' is required. Install it with: pip install packaging")
    sys.exit(1)

# --- Paste your requirements here (exactly as in requirements.txt) ---
REQUIREMENTS_TEXT = """
numpy==1.26.4
scipy==1.11.4
scikit-learn==1.3.2

torch==2.2.2
transformers==4.41.2
sentence-transformers==2.6.1

dask==2024.1.1
distributed==2024.1.1

yt-dlp==2024.7.16
youtube-transcript-api==0.6.2

langchain==0.1.20
langchain-community==0.0.38

ddgs==9.0.0
beautifulsoup4==4.12.3
requests>=2.31.0

tqdm==4.66.4
psutil>=5.9.0

bokeh>=2.4.2,<3.0
"""

# Map pip names → import module names
IMPORT_MAP = {
    "scikit-learn": "sklearn",
    "sentence-transformers": "sentence_transformers",
    "yt-dlp": "yt_dlp",
    "youtube-transcript-api": "youtube_transcript_api",
    "beautifulsoup4": "bs4",
    "langchain-community": "langchain_community",
}

def parse_requirements(text):
    reqs = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        reqs.append(Requirement(line))
    return reqs

def get_installed_version(dist_name):
    # Python 3.10+: importlib.metadata is stdlib
    try:
        from importlib.metadata import version, PackageNotFoundError
    except ImportError:
        from importlib_metadata import version, PackageNotFoundError  # fallback

    try:
        return version(dist_name)
    except Exception:
        return None

def satisfies(installed_ver, specifier):
    if installed_ver is None:
        return False
    try:
        return Version(installed_ver) in specifier
    except Exception:
        return False

def check():
    print("\n🔍 Checking environment\n")
    print(f"💻 Host: {socket.gethostname()}")
    print(f"🐍 Python: {sys.version.split()[0]}\n")

    reqs = parse_requirements(REQUIREMENTS_TEXT)

    all_ok = True

    for r in reqs:
        dist_name = r.name  # e.g., 'scikit-learn'
        import_name = IMPORT_MAP.get(dist_name, dist_name.replace("-", "_"))

        # 1) Try import
        try:
            importlib.import_module(import_name)
            import_ok = True
        except Exception:
            import_ok = False

        # 2) Check installed version
        installed_ver = get_installed_version(dist_name)

        # 3) Check specifier compliance (==, >=, <, etc.)
        spec = r.specifier  # e.g., '==1.26.4' or '>=2.31.0'
        if spec:
            version_ok = satisfies(installed_ver, spec)
        else:
            version_ok = installed_ver is not None

        # 4) Decide status
        if import_ok and version_ok:
            print(f"✔ {dist_name:<30} ({installed_ver})")
        elif import_ok and not version_ok:
            print(f"⚠ {dist_name:<30} version {installed_ver} NOT in '{spec}'")
            all_ok = False
        else:
            print(f"❌ {dist_name:<30} NOT INSTALLED")
            all_ok = False

    print("\n" + "=" * 60)
    if all_ok:
        print("✅ All requirements satisfied.")
    else:
        print("⚠️ Some requirements missing or incompatible.")
        print("   Run: pip install -r requirements.txt")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    check()