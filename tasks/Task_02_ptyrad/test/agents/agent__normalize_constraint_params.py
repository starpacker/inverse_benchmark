

# --- Extracted Dependencies ---

def _normalize_constraint_params(constraint_params):
    """Helper to convert old constraint param format."""
    normalized_params = {}
    for name, p in constraint_params.items():
        freq = p.get("freq", None)
        start_iter = p.get("start_iter", 1 if freq is not None else None)
        step = p.get("step", freq if freq is not None else 1)
        end_iter = p.get("end_iter", None)
        
        normalized_params[name] = {
            "start_iter": start_iter,
            "step": step,
            "end_iter": end_iter,
            **{k: v for k, v in p.items() if k not in ("freq", "step", "start_iter", "end_iter")},
        }
    return normalized_params
