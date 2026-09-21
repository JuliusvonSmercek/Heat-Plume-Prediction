"""Convert DaRUS heat-plume NPZ sims into pipeline RUN_*/pflotran.h5 layout.

Supports two raw layouts
------------------------
1. Flat / legacy ``step-1-single-heat-plume-npz``::

       <src>/Sim_*.npz  (or npz-files/, training_data/training_data_interimvelocities/)
       <src>/dataset_info_interimvelocities.yaml
       <src>/{temperature,normed_flow}_injection_series.{csv,npy}   (optional)

2. Packaged ``full_dataset-raw`` (DaRUS step-3 scaled domain)::

       <src>/training_data/Sim_*.npz
       <src>/general/dataset_info_interimvelocities.yaml
       <src>/general/{temperature,normed_flow}_injection_series.npy
       <src>/info.json   (optional DaRUS citation metadata)

Source channels (interimvelocities NPZ)
--------------------------------------
inputs[0] Conductivity [m/d]           -> Permeability X [m^2]
inputs[1] Hydraulic Head (t0) [m]      -> Liquid Pressure [Pa]
inputs[5] Wells: Max Flow Rates        -> Material ID (1=background, >=7=injection HP)
interim_labels[0/1] Darcy Vx/Vy [m/d]  -> Liquid X/Y-Velocity [m_per_y] at t=final_time
labels[0] Temperature (Summer) [C]     -> Temperature [C] at t=final_time

Initial velocities (inputs[3/4]) are stored at t=0 for completeness.

Axis reorientation
------------------
DaRUS NPZ arrays are shaped ``(nx, ny)`` with groundwater flow along Y.
Every field is transposed to ``(ny, nx)`` so the flow axis lies on array axis 0
(pipeline ``ncells[0]``). Velocity *components* are not swapped: DaRUS ``vy``
stays ``Liquid Y-Velocity``.

Also writes ``conversion_info.yaml`` under ``--out`` with duration, ambient /
spread, injection temperature & rate series, grid, and source paths.
"""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

import h5py
import numpy as np
import yaml

try:
    from code.injection_series_to_yaml import (
        DEFAULT_AMBIENT_C,
        DEFAULT_STEPS_PER_YEAR,
        NORMED_FLOW_SCALE,
        extract_year_cycle,
        load_series,
        pick_year_strongest_swing,
        steps_per_year_from_series,
    )
except ImportError:  # running as ``python code/convert_npz_dataset.py``
    from injection_series_to_yaml import (  # type: ignore
        DEFAULT_AMBIENT_C,
        DEFAULT_STEPS_PER_YEAR,
        NORMED_FLOW_SCALE,
        extract_year_cycle,
        load_series,
        pick_year_strongest_swing,
        steps_per_year_from_series,
    )

RHO_KG_M3 = 1000.0
G_M_S2 = 9.81
MU_PA_S = 1.0e-3
SECONDS_PER_DAY = 86400.0
DAYS_PER_YEAR = 365.25
random_seed = 2406
porosity = 0.25

TIME_INIT = "   0 Time  0.00000E+00 y"

INPUT_NAMES = [
    "Conductivity [m/d]",
    "Hydraulic Head (t0) [m a.s.l.]",
    "Aquifer Thickness (t0) [m]",
    "Darcy Velocity in x (t0) [m/d]",
    "Darcy Velocity in y (t0) [m/d]",
    "Wells: Max Flow Rates [m^3/d]",
]


def duration_years_from_info(info: dict) -> float:
    """Prefer last label timestep [d] / 365.25, rounded to a clean year count."""
    timesteps = info.get("timesteps") or []
    if timesteps:
        years = float(max(timesteps)) / DAYS_PER_YEAR
        # DaRUS seasonal labels end ~9.79 y → treat as 10 y for pipeline duration.
        return float(round(years)) if abs(years - round(years)) < 0.25 else years
    return 10.0


def time_pred_key(final_time_years: float) -> str:
    """PFLOTRAN-style HDF5 group name for the prediction snapshot."""
    return f"  20 Time  {final_time_years:0.5E} y"


def conductivity_to_permeability(k_m_per_d: np.ndarray) -> np.ndarray:
    """Hydraulic conductivity [m/d] -> intrinsic permeability [m^2]."""
    k_m_s = k_m_per_d / SECONDS_PER_DAY
    return k_m_s * MU_PA_S / (RHO_KG_M3 * G_M_S2)


def head_to_pressure(head_m: np.ndarray) -> np.ndarray:
    """Hydraulic head [m] -> liquid pressure [Pa] (ρ g h)."""
    return RHO_KG_M3 * G_M_S2 * head_m


def darcy_m_d_to_m_y(v_m_per_d: np.ndarray) -> np.ndarray:
    return v_m_per_d * DAYS_PER_YEAR


def wells_to_material_id(wells: np.ndarray) -> np.ndarray:
    """Map injection well cells (positive rate) to Material IDs (background=1, HPs>=7).

    DaRUS stores injection (+) and extraction (-) in the same channel. Extraction is
    omitted so step2 only seeds streamlines at injection holes. Shared pipeline
    Material ID logic is unchanged; this filter is converter-only.
    """
    material = np.ones(wells.shape, dtype=np.int32)
    ys, xs = np.where(wells > 0)
    for i, (y, x) in enumerate(zip(ys, xs, strict=True)):
        material[y, x] = 7 + i
    return material


def reorient_to_pipeline(field: np.ndarray) -> np.ndarray:
    """DaRUS ``(nx, ny)`` -> ``(ny, nx)`` with the flow axis on index 0."""
    return np.ascontiguousarray(field.T)


def write_pflotran_h5(
    out_path: Path,
    *,
    pressure: np.ndarray,
    permeability: np.ndarray,
    material_id: np.ndarray,
    vx0: np.ndarray,
    vy0: np.ndarray,
    vx_end: np.ndarray,
    vy_end: np.ndarray,
    temperature: np.ndarray,
    time_pred: str,
) -> None:
    """Write a minimal two-snapshot pflotran.h5 consumed by ``prepare_dataset``."""
    flat = {
        "pressure": np.ascontiguousarray(pressure, dtype=np.float64).ravel(),
        "permeability": np.ascontiguousarray(permeability, dtype=np.float64).ravel(),
        "material": np.ascontiguousarray(material_id, dtype=np.int32).ravel(),
        "vx0": np.ascontiguousarray(vx0, dtype=np.float64).ravel(),
        "vy0": np.ascontiguousarray(vy0, dtype=np.float64).ravel(),
        "vx_end": np.ascontiguousarray(vx_end, dtype=np.float64).ravel(),
        "vy_end": np.ascontiguousarray(vy_end, dtype=np.float64).ravel(),
        "temperature": np.ascontiguousarray(temperature, dtype=np.float64).ravel(),
        "vz": np.zeros(pressure.size, dtype=np.float64),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(out_path, "w") as f:
        for time_key, vx, vy, temp in (
            (TIME_INIT, flat["vx0"], flat["vy0"], flat["temperature"]),
            (time_pred, flat["vx_end"], flat["vy_end"], flat["temperature"]),
        ):
            g = f.create_group(time_key)
            g.create_dataset("Liquid Pressure [Pa]", data=flat["pressure"])
            g.create_dataset("Permeability X [m^2]", data=flat["permeability"])
            g.create_dataset("Material ID", data=flat["material"])
            g.create_dataset("Liquid X-Velocity [m_per_y]", data=vx)
            g.create_dataset("Liquid Y-Velocity [m_per_y]", data=vy)
            g.create_dataset("Liquid Z-Velocity [m_per_y]", data=flat["vz"])
            g.create_dataset("Temperature [C]", data=temp)


def convert_sim(npz_path: Path, run_dir: Path, *, time_pred: str, source_label: str) -> int:
    """Convert one NPZ; return number of injection Material ID cells."""
    data = np.load(npz_path)
    inputs = data["inputs"]
    labels = data["labels"]
    interim = data["interim_labels"]

    if inputs.shape[0] != 6:
        raise ValueError(f"{npz_path}: expected 6 input channels, got {inputs.shape}")

    conductivity, head, _thickness, vx0, vy0, wells = (reorient_to_pipeline(ch) for ch in inputs)
    vx_end, vy_end = (reorient_to_pipeline(ch) for ch in interim)
    temperature = reorient_to_pipeline(labels[0])  # Summer — strongest plume signal
    material_id = wells_to_material_id(wells)
    n_inject = int((material_id != 1).sum())

    write_pflotran_h5(
        run_dir / "pflotran.h5",
        pressure=head_to_pressure(head),
        permeability=conductivity_to_permeability(conductivity),
        material_id=material_id,
        vx0=darcy_m_d_to_m_y(vx0),
        vy0=darcy_m_d_to_m_y(vy0),
        vx_end=darcy_m_d_to_m_y(vx_end),
        vy_end=darcy_m_d_to_m_y(vy_end),
        temperature=temperature,
        time_pred=time_pred,
    )
    (run_dir / "pflotran.out").write_text(
        f"# Converted from {npz_path.name} ({source_label})\n", encoding="utf-8"
    )
    return n_inject


def write_settings(out_dir: Path, info: dict, *, final_time: float) -> dict:
    """Write pipeline settings.yaml files; return grid summary for conversion_info."""
    nx_darus = int(info["spatial domain"]["nx"])
    ny_darus = int(info["spatial domain"]["ny"])
    res_x_darus = float(info["spatial domain"]["res_x"])
    res_y_darus = float(info["spatial domain"]["res_y"])

    # After transpose: ncells[0] = ny (flow axis), ncells[1] = nx.
    nx = ny_darus
    ny = nx_darus
    res_x = res_y_darus
    res_y = res_x_darus
    size = [nx * res_x, ny * res_y, res_x]

    inputs_settings = {
        "general": {"dimensions": 2, "random_bool": False, "seed_id": random_seed},
        "grid": {"ncells": [nx, ny, 1], "size": size},
    }
    inputs_dir = out_dir / "inputs"
    inputs_dir.mkdir(parents=True, exist_ok=True)
    with open(inputs_dir / "settings.yaml", "w", encoding="utf-8") as f:
        yaml.dump(inputs_settings, f, default_flow_style=False, sort_keys=False)

    root_settings = {
        "general": {
            "output_directory": str(out_dir),
            "number_cells": [nx, ny, 1],
            "cell_resolution": res_x,
            "random_seed": random_seed,
            "time_to_simulate": {"final_time": final_time},
        },
        "hydrogeological_parameters": {
            "porosity": {"value": porosity},
        },
    }
    with open(out_dir / "settings.yaml", "w", encoding="utf-8") as f:
        yaml.dump(root_settings, f, default_flow_style=False, sort_keys=False)

    return {
        "darus_spatial_domain": dict(info["spatial domain"]),
        "pipeline_grid": {
            "ncells": [nx, ny, 1],
            "size_m": size,
            "cell_resolution_m": res_x,
            "note": "After transpose: axis0 = DaRUS Y (flow), axis1 = DaRUS X",
        },
    }


def resolve_dataset_info(src_dir: Path) -> tuple[dict, Path]:
    candidates = [
        src_dir / "general" / "dataset_info_interimvelocities.yaml",
        src_dir / "dataset_info_interimvelocities.yaml",
    ]
    for path in candidates:
        if path.is_file():
            with open(path, encoding="utf-8") as f:
                return yaml.safe_load(f), path
    raise FileNotFoundError(
        f"No dataset_info_interimvelocities.yaml under {src_dir} "
        f"(tried: {', '.join(str(p) for p in candidates)})"
    )


def discover_sims(src_dir: Path) -> list[tuple[int, Path]]:
    candidates = [
        src_dir / "training_data",
        src_dir / "npz-files",
        src_dir / "training_data" / "training_data_interimvelocities",
        src_dir,
    ]
    interim = next((p for p in candidates if p.is_dir() and any(p.glob("Sim_*.npz"))), None)
    if interim is None:
        raise FileNotFoundError(
            f"No Sim_*.npz found under {src_dir} "
            f"(tried: {', '.join(str(p) for p in candidates)})"
        )
    sims: list[tuple[int, Path]] = []
    for path in interim.glob("Sim_*.npz"):
        match = re.fullmatch(r"Sim_(\d+)\.npz", path.name)
        if match:
            sims.append((int(match.group(1)), path))
    return sorted(sims, key=lambda t: t[0])


def find_injection_series(src_dir: Path) -> tuple[Path | None, Path | None]:
    """Locate temperature / normed-flow series (``.npy`` preferred, then ``.csv``)."""
    search_dirs = [src_dir / "general", src_dir]
    temp_path = flow_path = None
    for directory in search_dirs:
        if not directory.is_dir():
            continue
        for stem, slot in (
            ("temperature_injection_series", "temp"),
            ("normed_flow_injection_series", "flow"),
        ):
            for ext in (".npy", ".csv", ".txt"):
                candidate = directory / f"{stem}{ext}"
                if candidate.is_file():
                    if slot == "temp" and temp_path is None:
                        temp_path = candidate
                    if slot == "flow" and flow_path is None:
                        flow_path = candidate
                    break
    return temp_path, flow_path


def export_series_csv(series: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out_path, series, fmt="%.10g")


def values_dict(times: np.ndarray, values: np.ndarray) -> dict[str, float]:
    return {f"{t:.5f}": float(v) for t, v in zip(times, values, strict=True)}


def load_darus_citation(src_dir: Path) -> dict | None:
    info_json = src_dir / "info.json"
    if not info_json.is_file():
        return None
    raw = json.loads(info_json.read_text(encoding="utf-8"))
    title = None
    try:
        fields = raw["metadataBlocks"]["citation"]["fields"]
        for field in fields:
            if field.get("typeName") == "title":
                title = field.get("value")
                break
    except (KeyError, TypeError):
        title = None
    return {
        "doi": raw.get("datasetPersistentId"),
        "title": title,
        "version_number": raw.get("versionNumber"),
        "publication_date": raw.get("publicationDate"),
        "license": (raw.get("license") or {}).get("name"),
    }


def build_injection_metadata(
    temp_path: Path,
    flow_path: Path,
    *,
    ambient_c: float,
    flow_scale: float,
    dt_days: float,
    year: int | str = "auto",
) -> tuple[dict, np.ndarray, np.ndarray]:
    temp_delta = load_series(temp_path)
    flow_norm = load_series(flow_path)
    n_steps = steps_per_year_from_series(temp_delta)
    year_idx = pick_year_strongest_swing(temp_delta, n_steps) if str(year).lower() == "auto" else int(year)
    times, temp_abs, rate = extract_year_cycle(
        temp_delta,
        flow_norm,
        year=year_idx,
        n_steps=n_steps,
        ambient_c=ambient_c,
        flow_scale=flow_scale,
    )
    spread = float(np.max(np.abs(temp_delta)))
    body_len = len(temp_delta) - 1 if np.isclose(temp_delta[0], temp_delta[-1]) else len(temp_delta)
    return {
        "source_temperature_series": str(temp_path),
        "source_normed_flow_series": str(flow_path),
        "series_length": int(len(temp_delta)),
        "series_layout": (
            f"{body_len // n_steps} years × {n_steps} samples/year"
            + (" + 1 wrap-around duplicate" if body_len + 1 == len(temp_delta) else "")
        ),
        "note": (
            "Injection series uses 73 samples/year; dataset_info "
            "'timeresolution [d]: 18.25' is for field snapshots only."
        ),
        "dt_days_field_snapshots": float(dt_days) if dt_days is not None else None,
        "steps_per_year": int(n_steps),
        "year_index": int(year_idx),
        "year_selection": "auto_strongest_delta_t_swing" if str(year).lower() == "auto" else "manual",
        "ambient_temperature_C": float(ambient_c),
        "temperature_spread_C": spread,
        "temperature_delta_C_range": [float(temp_delta.min()), float(temp_delta.max())],
        "normed_flow_scale": float(flow_scale),
        "normed_flow_range": [float(flow_norm.min()), float(flow_norm.max())],
        "injection_temperature_C": {
            "time_unit": "year",
            "values": values_dict(times, temp_abs),
            "absolute_C_range": [float(temp_abs.min()), float(temp_abs.max())],
        },
        "injection_rate_m3_per_s": {
            "time_unit": "year",
            "values": values_dict(times, rate),
            "range": [float(rate.min()), float(rate.max())],
        },
        "full_series": {
            "temperature_delta_C": temp_delta.tolist(),
            "normed_flow": flow_norm.tolist(),
        },
    }, temp_delta, flow_norm


def convert_dataset(
    src_dir: Path,
    out_dir: Path,
    *,
    sim_ids: list[int] | None = None,
    limit: int | None = None,
    clean: bool = False,
    ambient_c: float = DEFAULT_AMBIENT_C,
    flow_scale: float = NORMED_FLOW_SCALE,
    injection_year: str | int = "auto",
) -> list[int]:
    info, info_path = resolve_dataset_info(src_dir)
    final_time = duration_years_from_info(info)
    time_pred = time_pred_key(final_time)
    dt_days = float(info.get("timeresolution [d]", 18.25))

    if clean and out_dir.exists():
        for child in out_dir.iterdir():
            if child.is_dir() and (child.name.startswith("RUN_") or child.name in {"inputs", "general"}):
                shutil.rmtree(child)
            elif child.name in {"settings.yaml", "conversion_info.yaml"}:
                child.unlink()

    out_dir.mkdir(parents=True, exist_ok=True)
    grid_meta = write_settings(out_dir, info, final_time=final_time)

    temp_path, flow_path = find_injection_series(src_dir)
    injection_meta: dict | None = None
    if temp_path is not None and flow_path is not None:
        injection_meta, temp_delta, flow_norm = build_injection_metadata(
            temp_path,
            flow_path,
            ambient_c=ambient_c,
            flow_scale=flow_scale,
            dt_days=dt_days,
            year=injection_year,
        )
        general_out = out_dir / "general"
        export_series_csv(temp_delta, general_out / "temperature_injection_series.csv")
        export_series_csv(flow_norm, general_out / "normed_flow_injection_series.csv")
        np.save(general_out / "temperature_injection_series.npy", temp_delta)
        np.save(general_out / "normed_flow_injection_series.npy", flow_norm)
        injection_meta["exported_csv"] = {
            "temperature_delta_C": str(general_out / "temperature_injection_series.csv"),
            "normed_flow": str(general_out / "normed_flow_injection_series.csv"),
        }
        injection_meta["exported_npy"] = {
            "temperature_delta_C": str(general_out / "temperature_injection_series.npy"),
            "normed_flow": str(general_out / "normed_flow_injection_series.npy"),
        }
    else:
        print("Warning: injection series not found; conversion_info will omit schedules.")

    sims = discover_sims(src_dir)
    if sim_ids is not None:
        wanted = set(sim_ids)
        sims = [(i, p) for i, p in sims if i in wanted]
    if limit is not None:
        sims = sims[:limit]

    source_label = src_dir.name
    converted: list[int] = []
    injection_counts: dict[int, int] = {}
    for sim_id, path in sims:
        run_dir = out_dir / f"RUN_{sim_id}"
        if run_dir.exists():
            shutil.rmtree(run_dir)
        n_inject = convert_sim(path, run_dir, time_pred=time_pred, source_label=source_label)
        converted.append(sim_id)
        injection_counts[sim_id] = n_inject
        if len(converted) % 5 == 0 or len(converted) == len(sims):
            print(f"Converted {len(converted)}/{len(sims)} sims...")

    conversion_info = {
        "source": {
            "path": str(src_dir.resolve()),
            "dataset_info": str(info_path.resolve()),
            "citation": load_darus_citation(src_dir),
        },
        "conversion": {
            "output_path": str(out_dir.resolve()),
            "random_seed": random_seed,
            "porosity": porosity,
            "wells_policy": "injection_only_positive_rate",
            "label_channel": "Temperature (Summer) [C]",
            "final_time_years": final_time,
            "hdf5_time_prediction_key": time_pred,
            "converted_sim_ids": converted,
            "injection_well_counts": injection_counts,
        },
        "timing": {
            "timeresolution_d": dt_days,
            "timesteps_d": list(info.get("timesteps") or []),
            "duration_years": final_time,
            "label_times_years": [float(t) / DAYS_PER_YEAR for t in (info.get("timesteps") or [])],
        },
        "channels": {
            "inputs": info.get("inputs"),
            "interim_labels": info.get("interim_labels"),
            "labels": info.get("labels"),
        },
        "grid": grid_meta,
        "physical_parameters_for_settings": {
            "resolution_m": grid_meta["pipeline_grid"]["cell_resolution_m"],
            "duration_years": final_time,
            "porosity_frac": porosity,
            "ambient_temperature_C": ambient_c,
            "temperature_spread_C": (
                injection_meta["temperature_spread_C"] if injection_meta else None
            ),
            "normed_flow_scale": flow_scale,
        },
        "injection": injection_meta,
    }
    # Drop full_series from the nested copy used for settings preview size; keep in file
    # but write with default representers (lists are fine for 731 points).
    with open(out_dir / "conversion_info.yaml", "w", encoding="utf-8") as f:
        yaml.dump(conversion_info, f, default_flow_style=False, sort_keys=False, width=120)

    print(f"Done: {len(converted)} RUN folders in {out_dir}")
    print(f"Wrote {out_dir / 'conversion_info.yaml'} (duration={final_time} y)")
    return converted


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--src",
        type=Path,
        default=Path("datasets/full_dataset-raw"),
        help="Path to the raw DaRUS dataset root",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("datasets/full_dataset"),
        help="Output dataset root (RUN_* + settings.yaml + conversion_info.yaml)",
    )
    parser.add_argument("--ids", type=int, nargs="*", default=None, help="Optional Sim IDs to convert")
    parser.add_argument("--limit", type=int, default=None, help="Convert only the first N sims")
    parser.add_argument(
        "--clean",
        action="store_true",
        help="Remove existing RUN_* / inputs / general / settings in --out before converting",
    )
    parser.add_argument(
        "--ambient",
        type=float,
        default=DEFAULT_AMBIENT_C,
        help=f"Ambient temperature [°C] for ΔT series (default {DEFAULT_AMBIENT_C})",
    )
    parser.add_argument(
        "--flow-scale",
        type=float,
        default=NORMED_FLOW_SCALE,
        help=f"Normed flow → m³/s scale (default {NORMED_FLOW_SCALE})",
    )
    parser.add_argument(
        "--injection-year",
        default="auto",
        help="0-based calendar year for the 1 y injection cycle, or 'auto'",
    )
    args = parser.parse_args()
    convert_dataset(
        args.src,
        args.out,
        sim_ids=args.ids,
        limit=args.limit,
        clean=args.clean,
        ambient_c=args.ambient,
        flow_scale=args.flow_scale,
        injection_year=args.injection_year,
    )


if __name__ == "__main__":
    main()
