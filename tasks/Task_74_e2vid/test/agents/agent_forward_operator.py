import numpy as np

import matplotlib

matplotlib.use('Agg')

def forward_operator(log_intensity, events, event_idx, t_end, contrast_threshold):
    """
    Forward model: Apply event-based log-intensity updates.
    
    Given the current log-intensity state, integrate events up to t_end
    and return the predicted intensity image.
    
    Args:
        log_intensity: Current log-intensity state (height, width)
        events: List of events (x, y, t, polarity)
        event_idx: Current index in events list
        t_end: End time for integration
        contrast_threshold: Event triggering threshold C
    
    Returns:
        intensity: Predicted intensity image
        new_event_idx: Updated event index
    """
    log_int = log_intensity.copy()
    n_events = len(events)
    
    while event_idx < n_events and events[event_idx][2] < t_end:
        x, y, t, p = events[event_idx]
        x, y = int(x), int(y)
        log_int[y, x] += p * contrast_threshold
        event_idx += 1
    
    # Convert back to intensity
    intensity = np.exp(log_int)
    intensity = np.clip(intensity, 0, 5)
    
    return intensity, event_idx
