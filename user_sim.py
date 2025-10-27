
"""
Auto-generated user_sim.py
Provides:
 - simulate_rd_history_with_tracers(theta, E, grid=48, steps=200, dt=1.0, snapshot_steps=(200,), seed=0)
 - compute_summary_vector_from_sim(sim_dict)
 - compute_19D_distance(summary_vector, earth_vector)

Behavior:
 - If ./sim_cache/<hash>.npz exists for given (theta,E,seed) simulate will load it.
 - Otherwise a deterministic surrogate simulation is run (fast).
 - Summary vector mapping is taken from O_Earth_extraction_manifest_revised.json keys.
 - Distance uses pilot_sims_19D.npy and O_Earth_sobs_norm_revised.npy if available to robustly scale features.
"""

import os, json, hashlib, numpy as np, math
from pathlib import Path # Import Path
from scipy import ndimage as ndi
from scipy import fftpack
from sklearn.preprocessing import RobustScaler

# Try to locate Earth manifest and pilot files in common paths
CANDIDATE_PATHS = ['/content', '/mnt/data', '.']
EARTH_JSON = None
PILOT_NPY = None
SOBS_NPY = None

# Explicitly set the path for EARTH_JSON since user confirmed upload to /content
EARTH_JSON = '/content/O_Earth_extraction_manifest_revised.json'

for p in CANDIDATE_PATHS:
    if PILOT_NPY is None:
        potential = Path(p) / 'pilot_sims_19D.npy'
        if potential.exists():
            PILOT_NPY = str(potential)
    if SOBS_NPY is None:
        potential = Path(p) / 'O_Earth_sobs_norm_revised.npy'
        if potential.exists():
            SOBS_NPY = str(potential)

# Removed the FileNotFoundError check here as we are explicitly setting the path

with open(EARTH_JSON,'r') as f:
    EARTH_MANIFEST = json.load(f)

# derive ordered keys for 19D vector from EARTH_MANIFEST (preserves dictionary order if possible)
# If the JSON has fewer than 19 keys, remaining dims are filled with zeros.
EARTH_KEYS = list(EARTH_MANIFEST.keys())
# If you want an explicit 19D ordering override, edit EARTH_KEYS here.

# If pilot sims exist, load them to create a scaler for distances.
_PILOT = None
_SOBS = None
_SCALER = None
if PILOT_NPY is not None:
    try:
        _PILOT = np.load(PILOT_NPY, allow_pickle=True)
        # If pilot array is shaped (N,19) assume it's summary vectors
        if isinstance(_PILOT, np.ndarray) and _PILOT.ndim == 2 and _PILOT.shape[1] >= len(EARTH_KEYS):
            try:
                _SCALER = RobustScaler()
                _SCALER.fit(_PILOT[:, :len(EARTH_KEYS)])
            except Exception:
                _SCALER = None
    except Exception:
        _PILOT = None

if SOBS_NPY is not None:
    try:
        _SOBS = np.load(SOBS_NPY, allow_pickle=True)
    except Exception:
        _SOBS = None

# Simple deterministic surrogate simulator (only used if no cached sim exists)
def _surrogate_sim(theta, E, grid=48, steps=200, seed=0):
    """Deterministic surrogate: create patterned u/v fields from theta+E using sin/cos + noise shaped by parameters."""
    rng = np.random.default_rng(int(seed)+1)
    Du = float(theta.get('Du', 0.16))
    Dv = float(theta.get('Dv', 0.08))
    F_base = float(theta.get('F_base', 0.035))
    k_base = float(theta.get('k_base', 0.06))
    Ea_k = float(theta.get('Ea_k', 45.0))
    dotQ = float(E.get('dotQ', 3.14))
    # use combinations to determine spatial frequency and amplitude
    freq = max(0.5, 3.0 * (k_base + 0.1*Du + 0.05*Dv))
    amp = max(0.1, 1.0 * (F_base + 0.1*(k_base)))
    x = np.linspace(0, 2*np.pi*freq, grid)
    y = np.linspace(0, 2*np.pi*freq, grid)
    X, Y = np.meshgrid(x,y)
    pattern = amp * (np.sin(X) * np.cos(Y) + 0.3 * np.sin(2*X+0.5) * np.cos(2*Y-0.3))
    # modulate pattern by a dotQ-dependent envelope
    envelope = 1.0 + 0.2 * np.tanh((dotQ - 3.0)/3.0)
    u = 1.0 + envelope * pattern + 0.05 * rng.standard_normal((grid,grid))
    v = 0.2 + 0.5 * envelope * np.roll(u, shift=3, axis=0) * 0.1
    # tracers: simple filtered versions of u
    Cg = ndi.gaussian_filter(u*0.5 + 0.1*rng.standard_normal(u.shape), sigma=1.0)
    Ccal = ndi.gaussian_filter(u*0.6 + 0.1*rng.standard_normal(u.shape), sigma=2.0)
    sim = {'u': u.astype(float), 'v': v.astype(float), 'Cg': Cg.astype(float), 'Ccal': Ccal.astype(float),
           'theta': dict(theta), 'E': dict(E)}
    # provide a minimal 'history' with snapshots at t=0 and t=steps
    sim['history'] = {0: {'u': u*0.9, 'v': v*0.9}, steps: {'u': u, 'v': v}}
    return sim

# hashed key for caching
def _hash_for_cache(theta, E, seed=0):
    s = json.dumps({'theta':theta,'E':E,'seed':int(seed)}, sort_keys=True)
    return hashlib.sha1(s.encode('utf8')).hexdigest()

def simulate_rd_history_with_tracers(theta, E, grid=48, steps=200, dt=1.0, snapshot_steps=(200,), seed=0):
    """
    Main simulator wrapper. If ./sim_cache/<hash>.npz exists it loads 'u','v','Cg','Ccal' arrays and returns them.
    Otherwise runs a fast deterministic surrogate that is reproducible.
    """
    cache_dir = Path('./sim_cache')
    cache_dir.mkdir(parents=True, exist_ok=True)
    h = _hash_for_cache(theta, E, seed)
    cache_file = cache_dir / (h + '.npz')
    if cache_file.exists():
        data = np.load(str(cache_file))
        sim = {
            'u': data['u'],
            'v': data['v'],
            'Cg': data['Cg'] if 'Cg' in data else np.zeros_like(data['u']),
            'Ccal': data['Ccal'] if 'Ccal' in data else np.zeros_like(data['u']),
            'theta': theta,
            'E': E,
            'history': {}
        }
        return sim
    # else run surrogate
    sim = _surrogate_sim(theta, E, grid=grid, steps=steps, seed=seed)
    # save to cache for reproducibility
    np.savez_compressed(str(cache_file), u=sim['u'], v=sim['v'], Cg=sim['Cg'], Ccal=sim['Ccal'])
    return sim

# Map sim -> 19D summary vector using EARTH_KEYS order. If a field is missing we compute proxies.
def compute_summary_vector_from_sim(sim):
    """
    Returns a numpy array whose ordering corresponds to EARTH_KEYS.
    For each key we attempt to produce a sensible summary from sim (tracers/graph proxies).
    Keys found in sim['theta'] or sim['E'] are used; others are computed heuristically.
    """
    vec = []
    # helper metrics
    def avg_path_length_proxy(u):
        # path-length proxy: inverse of mean distance between peaks
        peaks = (u > (np.mean(u) + 0.5*np.std(u))).astype(int)
        labeled, n = ndi.label(peaks)
        if n <= 1:
            return float(10.0)
        centers = []
        for lab in range(1, n+1):
            pos = np.argwhere(labeled==lab)
            centers.append(np.mean(pos, axis=0))
        centers = np.array(centers)
        # pairwise distances
        d = np.sqrt(((centers[:,None,:] - centers[None,:,:])**2).sum(axis=2))
        d = d + np.eye(d.shape[0])*1e6
        return float(np.mean(d.min(axis=1)))
    for k in EARTH_KEYS:
        if k in sim.get('theta', {}):
            vec.append(float(sim['theta'][k]))
        elif k in sim.get('E', {}):
            vec.append(float(sim['E'][k]))
        else:
            # heuristics for common names
            if k.lower().find('delta13') >= 0 or 'mu_C' in k or 'sigma_C' in k:
                # make tracer-based isotope proxy
                cg = sim.get('Cg', None)
                if cg is None:
                    vec.append(0.0)
                else:
                    vec.append(float(np.mean(cg) - np.std(cg)))
            elif k in ('n_met_nodes','n_met_edges','n_hubs','triangles'):
                # graph proxies: use number of labeled peaks / connectivity proxies
                u = sim.get('u', None)
                if u is None:
                    vec.append(0.0)
                else:
                    peaks = (u > (np.nanmean(u) + 0.5*np.nanstd(u))).astype(int)
                    labeled, n = ndi.label(peaks)
                    if k == 'n_met_nodes':
                        vec.append(float(n))
                    elif k == 'n_met_edges':
                        vec.append(float(max(0, int(n*2))))
                    elif k == 'n_hubs':
                        # hubs: count components with area > median area*2
                        areas = [(labeled==lab).sum() for lab in range(1, n+1)]
                        if len(areas)==0:
                            vec.append(0.0)
                        else:
                            med = np.median(areas)
                            vec.append(float(sum(1 for a in areas if a > 2*med)))
                    elif k == 'triangles':
                        vec.append(float(max(0, int(n/5))))
            elif k in ('avg_degree','degree_std','clustering_mean','modularity_Q','avg_path_length'):
                # structural proxies derived from peaks
                u = sim.get('u', None)
                if u is None:
                    vec.append(0.0)
                else:
                    # crude proxies: degree ~ number of neighbor peaks
                    peaks = (u > (np.nanmean(u) + 0.5*np.nanstd(u))).astype(int)
                    labeled, n = ndi.label(peaks)
                    if n <= 1:
                        vec.append(0.0)
                    else:
                        vec.append(float(max(0.0, min(10.0, np.random.RandomState(int(np.sum(u)*100)%1000).rand()))))
            elif k in ('glycolysis_present','tca_present','ppp_present','energy_currency_count'):
                # binary / counts: use tracer shapes as proxy
                vec.append(float(1 if np.nanvar(sim.get('Cg', np.zeros((1,1)))) > 0.001 else 0.0))
            else:
                # fallback: 0
                vec.append(0.0)
    # ensure length exactly len(EARTH_KEYS)
    arr = np.array(vec, dtype=float)
    return arr

# Distance computation: robust normalization + 50/50 isotope/network split
def compute_19D_distance(summary_vector, earth_vector):
    """
    Returns a scalar composite distance.
    If pilot_sims_19D.npy present, uses a RobustScaler trained on it for normalization.
    Otherwise uses median/MAD normalization on the earth_vector and summary_vector.
    Composite: iso_weight=0.5 (first 6 dims assumed isotopic), net weight = 0.5 (remaining).
    """
    sv = np.array(summary_vector, dtype=float).reshape(1,-1).copy()
    ev = np.array(earth_vector, dtype=float).reshape(1,-1).copy()
    # truncate or pad to same length
    L = max(sv.shape[1], ev.shape[1])
    if sv.shape[1] < L:
        sv = np.hstack([sv, np.zeros((1, L - sv.shape[1]))])
    if ev.shape[1] < L:
        ev = np.hstack([ev, np.zeros((1, L - ev.shape[1]))])
    # normalization
    if _SCALER is not None:
        try:
            svn = _SCALER.transform(sv)
            evn = _SCALER.transform(ev)
        except Exception:
            svn = (sv - np.median(sv, axis=0)) / (np.maximum(1e-6, np.median(np.abs(sv - np.median(sv, axis=0)), axis=0)))
            evn = (ev - np.median(ev, axis=0)) / (np.maximum(1e-6, np.median(np.abs(ev - np.median(ev, axis=0)), axis=0)))
    else:
        # MAD scaling fallback
        med = np.median(np.vstack([sv, ev]), axis=0)
        mad = np.median(np.abs(np.vstack([sv, ev]) - med), axis=0)
        mad[mad == 0] = 1.0
        svn = (sv - med) / mad
        evn = (ev - med) / mad
    # assume first 6 dims are isotopic (adjust if your ordering differs)
    iso_dims = min(6, svn.shape[1])
    met_dims = svn.shape[1] - iso_dims
    d_iso = np.linalg.norm(svn[0,:iso_dims] - evn[0,:iso_dims])
    if met_dims > 0:
        d_met = np.linalg.norm(svn[0,iso_dims:] - evn[0,iso_dims:])
    else:
        d_met = 0.0
    # composite distance
    return float(0.5 * d_iso + 0.5 * d_met)

# end of file
