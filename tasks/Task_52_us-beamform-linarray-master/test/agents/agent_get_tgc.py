import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def get_tgc(alpha0, prop_dist, transmit_freq):
    """Time-gain compensation"""
    n = 1  # approx. 1 for soft tissue
    alpha = alpha0 * (transmit_freq * 1e-6)**n
    # tgc_gain = 10^(alpha * dist_cm / 20 * ???) 
    # Original formula: tgc_gain = 10**(alpha * prop_dist * 100 / 20)
    tgc_gain = 10**(alpha * prop_dist * 100 / 20)
    return tgc_gain
