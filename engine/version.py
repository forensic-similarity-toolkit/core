# engine/version.py

"""
FSIS Engine Version Module
--------------------------

Defines the official version of the FSIS engine.
This is used in all audit logs, reports, and Run ID generation.
"""

__version__ = "1.0.0"
__engine_name__ = "Forensic Similarity Intelligence System (FSIS)"


def get_version_info() -> dict:
    """
    Returns a dictionary containing engine name and version.
    Useful for embedding in reports and audit logs.
    """
    return {
        "engine_name": __engine_name__,
        "engine_version": __version__,
    }


if __name__ == "__main__":
    info = get_version_info()
    print(f"{info['engine_name']} — Version {info['engine_version']}")
