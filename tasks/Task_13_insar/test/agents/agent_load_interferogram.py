import numpy as np
import os
import collections


# --- Extracted Dependencies ---

def load_dem_rsc(filename, lower=False):
    """Loads and parses the .dem.rsc file"""
    RSC_KEY_TYPES = [
        ("width", int),
        ("file_length", int),
        ("x_first", float),
        ("y_first", float),
        ("x_step", float),
        ("y_step", float),
        ("x_unit", str),
        ("y_unit", str),
        ("z_offset", int),
        ("z_scale", int),
        ("projection", str),
    ]

    output_data = collections.OrderedDict()

    rsc_filename = (
        "{}.rsc".format(filename) if not filename.endswith(".rsc") else filename
    )
    with open(rsc_filename, "r") as f:
        for line in f.readlines():
            for field, num_type in RSC_KEY_TYPES:
                if line.startswith(field.upper()):
                    output_data[field] = num_type(line.split()[1])

    if lower:
        output_data = {k.lower(): d for k, d in output_data.items()}
    return output_data

def load_interferogram(filename, dtype=np.complex64, columns=None, rsc_file=None):
    """Load binary complex interferogram file."""
    data = np.fromfile(filename, dtype)
    if columns is None:
        if rsc_file is None:
            try:
                rsc_file = filename + ".rsc"
                if os.path.exists(rsc_file):
                    rsc_data = load_dem_rsc(rsc_file, lower=True)
                    columns = rsc_data["width"]
            except Exception:
                pass

    if columns is None:
        raise ValueError("Could not determine number of columns for interferogram.")

    return data.reshape((-1, columns))
