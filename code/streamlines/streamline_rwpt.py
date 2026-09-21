"""Physical RWPT thermal transport (seasonal injection -> temperature prior).

Advection comes entirely from the supplied velocity field. That field is the
end-state flow field and already contains the heat pumps, so no analytic radial well
term is added here -- doing so would place every well in the domain twice. The wells
enter this module only as energy sources.

Energy is deposited along the whole particle path rather than only at the final
snapshot. This is the frozen-field path equivalence: for a time-invariant transport
field, the ensemble injected at time s and observed at final time T has the same law
as the same path evaluated at elapsed time a = T - s. Summing one path over all ages
reconstructs the snapshot at T at ~1/T of the cost. The equivalence requires a
stationary field, which the end-state field satisfies by construction.

Formulation: sampled Darcy flux -> seepage velocity -> thermal velocity v_s/R,
alpha |v_s| / R mechanical dispersion, deposition via the particle's own analytic
transition kernel (see deposit_energy_gaussian).

Fixed-flux vs. maximum principle
--------------------------------
The wells are modelled as fixed-flux sources (Edot = Q rho_w c_w (T_inj - T_amb),
constant). A fixed-flux source obeys no maximum principle: in a field with no outflow
(recirculation, no extraction wells) energy is conserved but cannot leave, so a whole
region can be driven past the injection temperature. A real fixed-temperature
(Dirichlet) well self-limits -- as the surroundings warm toward T_inj the heat it can
deliver falls to zero.

We recover the Dirichlet answer with a per-cell acceptance field f(x) in [0, 1] that
scales every deposit landing in cell x: where a cell has already reached the injection
temperature it accepts no further energy. f is found by self-consistent iteration.

Multi-resolution acceptance solve
----------------------------------
A single global "worst cell over the whole domain" overshoot statistic, chased at one
fixed (fine) resolution from a blind a(x)=1 start, is a bad target when there are many
wells: whichever well's hot cell is currently worst gets suppressed, then a *different*
well's cell becomes the new worst, and the reported overshoot can sit at roughly the
same magnitude for many iterations while individual wells are, in fact, slowly
converging one at a time ("whack-a-mole"). It's also expensive and noisy to run at full
resolution when most of that resolution is irrelevant to the low-frequency shape of the
acceptance field.

Instead we solve the acceptance field on a sequence of increasingly fine spatial grids
(e.g. 10% -> 30% -> 100% of the full resolution), each level initialized by bilinearly
upsampling the previous (converged) level's acceptance field instead of starting over
from a(x)=1. At coarse resolution every well's near-field collapses onto a handful of
cells, which both concentrates the same particle budget into far fewer cells (much
better signal-to-noise per cell for the same cost) and makes "worst cell over the whole
domain" a far more informative, less whack-a-mole-prone statistic, since there are fewer
independent cells competing for that title. Each finer level then only has to refine a
warm-started field rather than discover the whole suppression pattern from scratch.
When a viz_dir is passed to run_rwpt_thermal_prior, each level dumps the coarse |q|
grid plus the warm-started acceptance, and each acceptance iteration dumps the
temperature prior, the acceptance used to produce it, and the signed overshoot.

Deposit kernel
--------------
Each particle's one-step displacement is drawn from a known anisotropic Gaussian (mean
= advective drift, std = sigmaL/sigmaT along/across the local flow direction) -- exactly
what drives the random walk itself. Rather than sampling one random position and
splatting a hard point there (bilinear Cloud-in-Cell), deposit_energy_gaussian deposits
that analytic kernel directly, centered at the pre-step position. This removes the
"which of 4 corners did this one random draw land in" shot noise without introducing
any smoothing beyond what the physics already implies, since the kernel width is the
same sigmaL/sigmaT already used to perturb the particle. The solver logs the largest
sigmaL/sigmaT it actually saw against the configured kernel radius so truncation (kernel
too small for the dispersion actually occurring) is visible in the run log rather than
silently biasing the result.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from tqdm import tqdm

from code.postprocessing.visualization import visualize_rwpt_fields
from code.utils import logging as log  # noqa: F401

# convert_injection_config tiles a 1-year periodic profile across the multi-year
# duration and linearly interpolates it onto the solve grid. Returns
# (values: float32 tensor of shape (timeSteps_count,), steps_per_year: int).
from code.utils.yaml_parser import convert_injection_config

# Physical constants (SI units)
SECONDS_PER_YEAR_S: float = 365.25 * 24.0 * 3600.0

# Particles are seeded on a ring one pixel from the well node rather than on the node
# itself: a radial source has (near-)zero interpolated velocity at its own centre, so a
# particle seeded exactly there would only leave by dispersion. One pixel puts the seed
# on the neighbouring nodes, where the outward flux is resolved.
SEED_RADIUS_PX: float = 1.0

# --------------------------------------------------------------------------------- #
# Acceptance-iteration defaults (overridable via mode_rwpt attributes)
# --------------------------------------------------------------------------------- #
# Sample count used for every grid level except the final (full-resolution) one.
DEFAULT_COARSE_SAMPLES: int = 10_000
# Break a stage once the worst overshoot beyond the physical band drops below this (C).
# Larger -> stop sooner (cheaper, looser); smaller -> iterate longer chasing the bound.
DEFAULT_CONVERGENCE_THRESHOLD_C: float = 0.05
# Under-relaxation of the acceptance update in (0, 1]. <1 damps the upstream->downstream
# oscillation (suppressing a cell cools everything downstream of it). 0.7 is stable.
DEFAULT_ACCEPTANCE_RELAXATION: float = 0.7
# Gaussian blur (px, in that level's own pixel units) applied to the acceptance field so
# its suppressed/unsuppressed boundary isn't ragged -- the true acceptance field is smooth.
DEFAULT_ACCEPTANCE_BLUR_PX: float = 1.0

# Multi-resolution acceptance solve: grid size at each level, as a fraction of the full
# (fine) grid in each dimension, and the max self-consistent iterations to spend at each
# level before moving on (last entry applies to the full-resolution level). If
# acceptance_grid_max_iters is shorter than acceptance_grid_fractions, its last value is
# repeated to fill the remaining levels.
# DEFAULT_ACCEPTANCE_GRID_FRACTIONS: list[float] = [0.030625, 0.06125, 0.125, 0.25, 0.5, 1.0]
# DEFAULT_ACCEPTANCE_GRID_MAX_ITERS: list[int] = [32, 16, 8, 4, 2, 1]
DEFAULT_ACCEPTANCE_GRID_FRACTIONS: list[float] = [1.0]
DEFAULT_ACCEPTANCE_GRID_MAX_ITERS: list[int] = [1]

# Deposit kernel: how many sigma-widths of patch to materialize per particle per step.
# Larger radius captures more of the true tail (less truncation bias) but costs
# (2r+1)^2 scatter-add work per particle per step instead of CIC's fixed 4.
#
# sigmaL/sigmaT (in PIXELS) scale as 1/dx for the same physical dispersion width, so a
# radius tuned for the full-resolution grid silently truncates at every coarser
# multi-resolution level, which uses physically LARGER pixels -> physically the same
# dispersion spreads over MORE pixels. Rather than picking one fixed radius (and
# manually re-tuning it whenever the grid or resolution changes), the driver computes
# it automatically per level -- see _auto_deposit_kernel_radius_px -- from the same
# dispersion physics local_transport_state uses, evaluated at the field's fastest
# seepage velocity (worst case, since mechanical dispersion grows with |v_s|) and that
# level's own dt. deposit_kernel_radius_px (below/in yaml) becomes the MAX radius: a
# cost ceiling exactly like the yaml `steps` value is for _min_steps_for_courant, never
# exceeded but only used as-is when the physics actually needs that much. The per-run
# deposit-kernel QA log line still reports what the widest step actually needed, so
# truncation against this ceiling (rather than a badly guessed fixed radius) is visible.
DEFAULT_MIN_DEPOSIT_KERNEL_RADIUS_PX: int = 2
DEFAULT_DEPOSIT_KERNEL_RADIUS_PX: int = 8

# Target thermal Courant number (Co = |v_T|_max * dt / dx) used to pick the step count
# automatically. Co <= 1 means particles don't cross more than one cell per step near
# the fastest-flowing wells; raise this only if you deliberately want to allow more
# per-step smearing there in exchange for fewer steps.
DEFAULT_TARGET_COURANT: float = 1.0


@dataclass
class RwptConfig:
    """Configuration for physical RWPT thermal transport simulation."""

    device: str
    resolution_m_per_px: float
    ambientTemp_C: float

    # Injection data: [Sources, TimeSteps]
    injectionRate_m3_per_s: torch.Tensor
    injectionTemp_C: torch.Tensor

    # Solve timeline
    timeSteps_count: int
    timeEnd_years: float
    seasonalCycleSteps_count: int

    # Sampling
    samplesPerSource_count: int

    # Aquifer properties
    porosity_frac: float
    rockDensity_kg_per_m3: float
    rockSpecificHeat_J_per_kgK: float
    waterDensity_kg_per_m3: float
    waterSpecificHeat_J_per_kgK: float
    thermalConductivityDry_W_per_mK: float
    thermalConductivityWet_W_per_mK: float
    thicknessAquifer_m: float

    # Dispersion parameters
    longitudinalDispersivity_m: float
    transverseDispersivityH_m: float

    # Memory batching constraint
    particles_per_batch_count: int

    # Deposit kernel radius (px, in THIS config's own pixel units -- coarser grid
    # levels have physically larger pixels, so the same radius covers a proportionally
    # larger physical footprint automatically).
    depositKernelRadius_px: int

    @property
    def timeStep_s(self) -> float:
        # dt = t_end * S_yr / N_steps
        return (self.timeEnd_years * SECONDS_PER_YEAR_S) / float(self.timeSteps_count)

    @property
    def volumetricHeatCapacityAquifer_J_per_m3K(self) -> float:
        # (rho c)_aq = theta rho_w c_w + (1-theta) rho_r c_r
        rho_cw_J_per_m3K = self.waterDensity_kg_per_m3 * self.waterSpecificHeat_J_per_kgK
        rho_cr_J_per_m3K = self.rockDensity_kg_per_m3 * self.rockSpecificHeat_J_per_kgK
        return self.porosity_frac * rho_cw_J_per_m3K + (1.0 - self.porosity_frac) * rho_cr_J_per_m3K

    @property
    def retardationFactor_dimless(self) -> float:
        # R = (rho c)_aq / (theta rho_w c_w)
        rho_cw_J_per_m3K = self.waterDensity_kg_per_m3 * self.waterSpecificHeat_J_per_kgK
        return self.volumetricHeatCapacityAquifer_J_per_m3K / (self.porosity_frac * rho_cw_J_per_m3K)

    @property
    def thermalDiffusivity_m2_per_s(self) -> float:
        # Conduction term of the retarded ADE, obtained by dividing the bulk energy
        # equation by (rho c)_aq:  D_cond = lambda_wet / (rho c)_aq. No further 1/R.
        return self.thermalConductivityWet_W_per_mK / self.volumetricHeatCapacityAquifer_J_per_m3K


@torch.jit.script
def sample_velocity_bilinear(
    positions_px: torch.Tensor,
    fieldTensor_m_per_year: torch.Tensor,
    invWidth_per_px: float,
    invHeight_per_px: float,
    resolution_m_per_px: float,
    secondsPerYear_s: float,
) -> torch.Tensor:
    """
    Interpolates the Darcy flux at particle positions.
    fieldTensor_m_per_year must be shape (1, 2, H, W); channel 0 is qx, channel 1 is qy.
    Returns tensor of shape (N, 2) in px/s with columns (qx, qy).
    """
    num_particles_count = positions_px.size(0)

    # align_corners=True mapping: xhat = 2x/(W-1) - 1 in [-1, 1]
    norm_x_dimless = positions_px[:, 0] * invWidth_per_px - 1.0
    norm_y_dimless = positions_px[:, 1] * invHeight_per_px - 1.0
    grid_dimless = torch.stack((norm_x_dimless, norm_y_dimless), dim=-1).view(1, num_particles_count, 1, 2)

    v_sample_m_per_year = F.grid_sample(
        fieldTensor_m_per_year,
        grid_dimless,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )

    # q_px = q_m/yr / (S_yr dx); sample is (1, 2, N, 1)
    v_m_per_s = v_sample_m_per_year.squeeze(0).squeeze(-1).t() / secondsPerYear_s
    return v_m_per_s / resolution_m_per_px


@torch.jit.script
def sample_acceptance_bilinear(
    positions_px: torch.Tensor,
    acceptanceField_1_1_hw: torch.Tensor,
    invWidth_per_px: float,
    invHeight_per_px: float,
) -> torch.Tensor:
    """Bilinearly sample the per-cell acceptance factor at particle positions.

    acceptanceField_1_1_hw must be shape (1, 1, H, W) with values in [0, 1].
    Returns a (N,) tensor of per-particle acceptance in [0, 1].
    """
    num_particles_count = positions_px.size(0)
    norm_x_dimless = positions_px[:, 0] * invWidth_per_px - 1.0
    norm_y_dimless = positions_px[:, 1] * invHeight_per_px - 1.0
    grid_dimless = torch.stack((norm_x_dimless, norm_y_dimless), dim=-1).view(1, num_particles_count, 1, 2)

    a_sample = F.grid_sample(
        acceptanceField_1_1_hw,
        grid_dimless,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return a_sample.view(num_particles_count)


@torch.jit.script
def deposit_energy_cic(
    grid_flat_J: torch.Tensor,
    positions_px: torch.Tensor,
    weights_J: torch.Tensor,
    active_mask_bool: torch.Tensor,
    width_px: int,
    height_px: int,
) -> None:
    """Bilinear Cloud-in-Cell deposition; fixed-shape scatters only.

    Kept for reference and unit-test coverage (see verify_rwpt_physical_validity, test
    3). The live solver deposits with deposit_energy_gaussian instead -- see the module
    docstring for why.
    """
    mask_bool = active_mask_bool & (weights_J.abs() > 1e-12)
    if not mask_bool.any():
        return

    pos_px = positions_px[mask_bool]
    w_J = weights_J[mask_bool]

    px_val = pos_px[:, 0]
    py_val = pos_px[:, 1]

    x0_px = torch.floor(px_val).to(torch.long)
    y0_px = torch.floor(py_val).to(torch.long)
    x1_px = x0_px + 1
    y1_px = y0_px + 1

    fx_dimless = px_val - x0_px.to(px_val.dtype)
    fy_dimless = py_val - y0_px.to(py_val.dtype)

    # w_ij = w (1-fx)^(1-i) fx^i (1-fy)^(1-j) fy^j, sum_ij w_ij = w
    w00_J = w_J * (1.0 - fx_dimless) * (1.0 - fy_dimless)
    w10_J = w_J * fx_dimless * (1.0 - fy_dimless)
    w01_J = w_J * (1.0 - fx_dimless) * fy_dimless
    w11_J = w_J * fx_dimless * fy_dimless

    m00_bool = (x0_px >= 0) & (x0_px < width_px) & (y0_px >= 0) & (y0_px < height_px)
    if m00_bool.any():
        idx00_idx = y0_px[m00_bool] * width_px + x0_px[m00_bool]
        grid_flat_J.scatter_add_(0, idx00_idx, w00_J[m00_bool])

    m10_bool = (x1_px >= 0) & (x1_px < width_px) & (y0_px >= 0) & (y0_px < height_px)
    if m10_bool.any():
        idx10_idx = y0_px[m10_bool] * width_px + x1_px[m10_bool]
        grid_flat_J.scatter_add_(0, idx10_idx, w10_J[m10_bool])

    m01_bool = (x0_px >= 0) & (x0_px < width_px) & (y1_px >= 0) & (y1_px < height_px)
    if m01_bool.any():
        idx01_idx = y1_px[m01_bool] * width_px + x0_px[m01_bool]
        grid_flat_J.scatter_add_(0, idx01_idx, w01_J[m01_bool])

    m11_bool = (x1_px >= 0) & (x1_px < width_px) & (y1_px >= 0) & (y1_px < height_px)
    if m11_bool.any():
        idx11_idx = y1_px[m11_bool] * width_px + x1_px[m11_bool]
        grid_flat_J.scatter_add_(0, idx11_idx, w11_J[m11_bool])


@torch.jit.script
def deposit_energy_gaussian(
    grid_flat_J: torch.Tensor,
    positions_px: torch.Tensor,
    weights_J: torch.Tensor,
    active_mask_bool: torch.Tensor,
    uL_dimless: torch.Tensor,
    sigmaL_px: torch.Tensor,
    sigmaT_px: torch.Tensor,
    width_px: int,
    height_px: int,
    kernel_radius_px: int,
) -> None:
    """Deposit energy as the particle's own known transition kernel instead of a point.

    A particle's one-step displacement is drawn from an anisotropic Gaussian (mean = the
    advective drift, std = sigmaL/sigmaT along/across the local flow direction) -- exactly
    what already drives the random walk. Depositing that analytic kernel directly, instead
    of drawing one random position and splatting a hard point there, removes the "which of
    4 corners did this one random draw land in" noise without introducing any smoothing
    that isn't already implied by the physics: the kernel width is the same sigmaL/sigmaT
    used to perturb the particle, not an arbitrary blur radius.

    The kernel is truncated to a (2*kernel_radius_px+1)^2 patch around the particle's
    pre-step position and renormalized to sum to 1 over its in-bounds cells, so total
    deposited energy still equals total weight exactly (same invariant as bilinear CIC)
    even though the physical tail beyond kernel_radius_px is folded back in rather than
    lost -- pick kernel_radius_px to comfortably cover the sigmas you actually see.
    """
    mask_bool = active_mask_bool & (weights_J.abs() > 1e-12)
    if not mask_bool.any():
        return

    device = grid_flat_J.device
    pos_px = positions_px[mask_bool]
    w_J = weights_J[mask_bool]
    uL_sel = uL_dimless[mask_bool]
    sigL_px = sigmaL_px[mask_bool].clamp(min=1e-6)
    sigT_px = sigmaT_px[mask_bool].clamp(min=1e-6)

    num_sel = pos_px.size(0)
    uLx = uL_sel[:, 0].view(num_sel, 1, 1)
    uLy = uL_sel[:, 1].view(num_sel, 1, 1)
    uTx = (-uL_sel[:, 1]).view(num_sel, 1, 1)
    uTy = uL_sel[:, 0].view(num_sel, 1, 1)

    px_val = pos_px[:, 0]
    py_val = pos_px[:, 1]

    base_x_idx = torch.floor(px_val).to(torch.long) - kernel_radius_px
    base_y_idx = torch.floor(py_val).to(torch.long) - kernel_radius_px

    patch_size = 2 * kernel_radius_px + 1
    offsets_idx = torch.arange(patch_size, device=device)
    row_offset_idx = offsets_idx.view(1, patch_size, 1)
    col_offset_idx = offsets_idx.view(1, 1, patch_size)

    gy_idx = base_y_idx.view(num_sel, 1, 1) + row_offset_idx
    gx_idx = base_x_idx.view(num_sel, 1, 1) + col_offset_idx

    dx_px = gx_idx.to(torch.float32) - px_val.view(num_sel, 1, 1)
    dy_px = gy_idx.to(torch.float32) - py_val.view(num_sel, 1, 1)

    # Project the offset onto the local (longitudinal, transverse) flow frame -- same
    # frame local_transport_state already builds for the random-walk draw itself.
    dL_px = dx_px * uLx + dy_px * uLy
    dT_px = dx_px * uTx + dy_px * uTy

    density = torch.exp(
        -0.5 * ((dL_px / sigL_px.view(num_sel, 1, 1)) ** 2 + (dT_px / sigT_px.view(num_sel, 1, 1)) ** 2)
    )

    in_bounds_bool = (gx_idx >= 0) & (gx_idx < width_px) & (gy_idx >= 0) & (gy_idx < height_px)
    density = torch.where(in_bounds_bool, density, torch.zeros_like(density))

    # Renormalize per particle over the (possibly truncated) in-bounds patch so total
    # deposited energy equals total weight exactly, same conservation invariant as CIC.
    norm = density.sum(dim=[1, 2], keepdim=True).clamp(min=1e-20)
    weight_patch_J = w_J.view(num_sel, 1, 1) * density / norm

    flat_idx = (gy_idx.clamp(0, height_px - 1) * width_px + gx_idx.clamp(0, width_px - 1)).view(-1)
    flat_weight_J = weight_patch_J.view(-1)
    grid_flat_J.scatter_add_(0, flat_idx, flat_weight_J)


@torch.jit.script
def local_transport_state(
    positions_px: torch.Tensor,
    fieldTensor_m_per_year: torch.Tensor,
    invWidth_per_px: float,
    invHeight_per_px: float,
    resolution_m_per_px: float,
    secondsPerYear_s: float,
    invPorosity_dimless: float,
    thermalRetardationScale_dimless: float,
    thermalDiffusivity_m2_per_s: float,
    alphaL_m: float,
    alphaT_m: float,
    res_sq_m2: float,
    uL_default_dimless: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    # The sampled field is the total specific discharge (Darcy flux) and already
    # includes every well; nothing is superposed on top of it here.
    q_px_per_s = sample_velocity_bilinear(
        positions_px,
        fieldTensor_m_per_year,
        invWidth_per_px,
        invHeight_per_px,
        resolution_m_per_px,
        secondsPerYear_s,
    )

    # v_s = q / theta (pore velocity), v_T = v_s / R (retarded thermal front)
    vSeepage_px_per_s = q_px_per_s * invPorosity_dimless
    vSeepageMag_px_per_s = torch.norm(vSeepage_px_per_s, dim=1, keepdim=True)
    vSeepageMag_m_per_s = vSeepageMag_px_per_s * resolution_m_per_px
    vThermal_px_per_s = vSeepage_px_per_s * thermalRetardationScale_dimless

    uL_dimless = torch.where(
        vSeepageMag_px_per_s > 1e-12,
        vSeepage_px_per_s / vSeepageMag_px_per_s,
        uL_default_dimless,
    )
    uT_dimless = torch.stack((-uL_dimless[:, 1], uL_dimless[:, 0]), dim=1)

    # Mechanical dispersion is alpha |v_s| on the pore velocity, retarded by 1/R only:
    # the theta in the bulk dispersive flux theta rho_w c_w D_mech cancels against the
    # theta in (rho c)_aq = theta rho_w c_w R.
    # D_L = (D_cond + alpha_L |v_s| / R) / dx^2
    # D_T = (D_cond + alpha_T |v_s| / R) / dx^2
    diffL_px2_per_s = (
        thermalDiffusivity_m2_per_s
        + alphaL_m * vSeepageMag_m_per_s * thermalRetardationScale_dimless
    ) / res_sq_m2
    diffT_px2_per_s = (
        thermalDiffusivity_m2_per_s
        + alphaT_m * vSeepageMag_m_per_s * thermalRetardationScale_dimless
    ) / res_sq_m2

    return vThermal_px_per_s, diffL_px2_per_s, diffT_px2_per_s, uL_dimless, uT_dimless


@torch.jit.script
def _rwpt_stream_step(
    accumEnergyGrid_flat_J: torch.Tensor,
    positions_px: torch.Tensor,
    fieldTensor_m_per_year: torch.Tensor,
    acceptanceField_1_1_hw: torch.Tensor,
    sourceIndices_idx: torch.Tensor,
    injectionEnergyRate_J_per_s: torch.Tensor,
    birthIndices_count: torch.Tensor,
    still_in_domain_bool: torch.Tensor,
    maxSigmaL_px_running: torch.Tensor,
    maxSigmaT_px_running: torch.Tensor,
    i_idx: int,
    steps_count: int,
    timeStep_s: float,
    width_px: int,
    height_px: int,
    resolution_m_per_px: float,
    weight_scale_s: float,
    invWidth_per_px: float,
    invHeight_per_px: float,
    res_sq_m2: float,
    thermalRetardationScale_dimless: float,
    invPorosity_dimless: float,
    sqrt_2_dt_s05: float,
    alphaL_m: float,
    alphaT_m: float,
    thermalDiffusivity_m2_per_s: float,
    secondsPerYear_s: float,
    uL_default_dimless: torch.Tensor,
    depositKernelRadius_px: int,
) -> None:
    """One timestep of the RWPT kernel: deposit (via the analytic transition kernel),
    predictor-corrector advance, absorbing-boundary latch.

    Split out of rwpt_seasonal_stream_kernel so the outer Python loop can drive it and
    report progress -- a torch.jit.script function can't hold a tqdm instance. In
    substance this is unchanged from a single-function scripted kernel, just
    parameterized per call (positions_px and still_in_domain_bool are mutated in place,
    same as before) instead of closed over an internal for-loop.

    maxSigmaL_px_running / maxSigmaT_px_running are single-element tensors the caller
    owns; this function bumps them (in place, via torch.maximum) with the largest
    per-step dispersion sigma actually seen among ACTIVE (currently depositing)
    particles, so the caller can log how well the deposit kernel radius covered the
    dispersion actually occurring in this run.
    """
    num_particles_count = positions_px.size(0)
    device = positions_px.device

    born_bool = i_idx >= birthIndices_count
    active_mask_bool = born_bool & still_in_domain_bool

    if not active_mask_bool.any():
        return

    # Deposit before stepping: the particle has travelled exactly age steps here,
    # which is the travel time the phase label assumes.
    # a = i - b, injection step s = T - 1 - a
    # For a not-yet-born particle (i_idx < birthIndices_count) age is negative, which
    # pushes injection_step_idx past steps_count-1 -- out of bounds for
    # injectionEnergyRate_J_per_s's time axis. active_mask_bool excludes these particles
    # from the deposit itself (in deposit_energy_gaussian, below), but that mask is
    # applied AFTER this gather, so the index must be clamped into range here regardless
    # of activity: the gathered value for an inactive particle is discarded later, it just
    # can't be allowed to read out of bounds first (silently wrong on CPU, a device-side
    # assert on CUDA).
    age_steps_count = i_idx - birthIndices_count
    injection_step_idx = (steps_count - 1 - age_steps_count).clamp(0, steps_count - 1)
    current_weight_J = injectionEnergyRate_J_per_s[sourceIndices_idx, injection_step_idx] * weight_scale_s

    # Per-cell acceptance: scale the deposit by how much energy the local cell can still
    # take. Where a cell has reached the injection temperature its acceptance is ~0, so
    # the fixed-flux source stops over-depositing there -- this is what turns the
    # (unbounded) Neumann source into the (bounded) Dirichlet well. The suppressed
    # energy simply is not deposited; it is not carried downstream, which is correct (a
    # well whose surroundings are at T_inj injects less net heat).
    accept_dimless = sample_acceptance_bilinear(
        positions_px, acceptanceField_1_1_hw, invWidth_per_px, invHeight_per_px
    )
    current_weight_J = current_weight_J * accept_dimless

    # Local transport state at the pre-step position is needed both to size the deposit
    # kernel (it IS this step's known transition distribution) and to drive the
    # predictor-corrector advance below, so it's computed once, ahead of the deposit.
    v0_px_per_s, dL0_px2_per_s, dT0_px2_per_s, uL0_dimless, uT0_dimless = local_transport_state(
        positions_px,
        fieldTensor_m_per_year,
        invWidth_per_px,
        invHeight_per_px,
        resolution_m_per_px,
        secondsPerYear_s,
        invPorosity_dimless,
        thermalRetardationScale_dimless,
        thermalDiffusivity_m2_per_s,
        alphaL_m,
        alphaT_m,
        res_sq_m2,
        uL_default_dimless,
    )
    sigmaL0_px = torch.sqrt(torch.clamp(dL0_px2_per_s, min=0.0)) * sqrt_2_dt_s05
    sigmaT0_px = torch.sqrt(torch.clamp(dT0_px2_per_s, min=0.0)) * sqrt_2_dt_s05

    active_sigmaL_px = sigmaL0_px[active_mask_bool]
    active_sigmaT_px = sigmaT0_px[active_mask_bool]
    maxSigmaL_px_running.copy_(torch.maximum(maxSigmaL_px_running, active_sigmaL_px.max().view(1)))
    maxSigmaT_px_running.copy_(torch.maximum(maxSigmaT_px_running, active_sigmaT_px.max().view(1)))

    deposit_energy_gaussian(
        accumEnergyGrid_flat_J,
        positions_px,
        current_weight_J,
        active_mask_bool,
        uL0_dimless,
        sigmaL0_px,
        sigmaT0_px,
        width_px,
        height_px,
        depositKernelRadius_px,
    )

    # Same noise draw is reused by predictor and corrector (Heun). The div(D) drift
    # correction for spatially varying D is not included; with the well now resolved
    # on the grid, D varies over a cell rather than singularly, so the residual bias
    # is small but nonzero near the source.
    noiseL_dimless = torch.randn((num_particles_count, 1), device=device)
    noiseT_dimless = torch.randn((num_particles_count, 1), device=device)

    disp_pred_px = (
        (v0_px_per_s * timeStep_s)
        + (uL0_dimless * noiseL_dimless * sigmaL0_px)
        + (uT0_dimless * noiseT_dimless * sigmaT0_px)
    )
    positions_star_px = positions_px + disp_pred_px

    v1_px_per_s, dL1_px2_per_s, dT1_px2_per_s, uL1_dimless, uT1_dimless = local_transport_state(
        positions_star_px,
        fieldTensor_m_per_year,
        invWidth_per_px,
        invHeight_per_px,
        resolution_m_per_px,
        secondsPerYear_s,
        invPorosity_dimless,
        thermalRetardationScale_dimless,
        thermalDiffusivity_m2_per_s,
        alphaL_m,
        alphaT_m,
        res_sq_m2,
        uL_default_dimless,
    )

    # X_{n+1} = X_n + 0.5(v0 + v1)dt + u_L xiL sigmaL + u_T xiT sigmaT
    v_mid_px_per_s = 0.5 * (v0_px_per_s + v1_px_per_s)
    dL_mid_px2_per_s = 0.5 * (dL0_px2_per_s + dL1_px2_per_s)
    dT_mid_px2_per_s = 0.5 * (dT0_px2_per_s + dT1_px2_per_s)
    uL_mid_dimless = 0.5 * (uL0_dimless + uL1_dimless)
    uL_mid_norm_dimless = torch.norm(uL_mid_dimless, dim=1, keepdim=True)
    uL_mid_dimless = torch.where(
        uL_mid_norm_dimless > 1e-12, uL_mid_dimless / uL_mid_norm_dimless, uL_default_dimless
    )
    uT_mid_dimless = torch.stack((-uL_mid_dimless[:, 1], uL_mid_dimless[:, 0]), dim=1)

    sigmaL_px = torch.sqrt(torch.clamp(dL_mid_px2_per_s, min=0.0)) * sqrt_2_dt_s05
    sigmaT_px = torch.sqrt(torch.clamp(dT_mid_px2_per_s, min=0.0)) * sqrt_2_dt_s05
    disp_px = (
        (v_mid_px_per_s * timeStep_s)
        + (uL_mid_dimless * noiseL_dimless * sigmaL_px)
        + (uT_mid_dimless * noiseT_dimless * sigmaT_px)
    )

    mask_expanded_bool = active_mask_bool.unsqueeze(1)
    positions_px.copy_(torch.where(mask_expanded_bool, positions_px + disp_px, positions_px))

    in_bounds_bool = (
        (positions_px[:, 0] >= 0.0)
        & (positions_px[:, 0] <= float(width_px - 1))
        & (positions_px[:, 1] >= 0.0)
        & (positions_px[:, 1] <= float(height_px - 1))
    )
    still_in_domain_bool.copy_(still_in_domain_bool & in_bounds_bool)


def rwpt_seasonal_stream_kernel(
    accumEnergyGrid_flat_J: torch.Tensor,
    positions_px: torch.Tensor,
    fieldTensor_m_per_year: torch.Tensor,
    acceptanceField_1_1_hw: torch.Tensor,
    sourceIndices_idx: torch.Tensor,
    injectionEnergyRate_J_per_s: torch.Tensor,
    birthIndices_count: torch.Tensor,
    timeStep_s: float,
    steps_count: int,
    width_px: int,
    height_px: int,
    resolution_m_per_px: float,
    alphaL_m: float,
    alphaT_m: float,
    thermalDiffusivity_m2_per_s: float,
    porosity_frac: float,
    retardationFactor_dimless: float,
    secondsPerYear_s: float,
    totalSamplesPerSource_count: int,
    depositKernelRadius_px: int = DEFAULT_DEPOSIT_KERNEL_RADIUS_PX,
    progress_desc: str = "RWPT solver steps",
) -> None:
    """Python-level driver over the `steps_count` timesteps.

    The heavy per-step math lives in `_rwpt_stream_step`, which is torch.jit.script'd
    for speed. Iteration itself happens here, in plain Python, only so a tqdm bar can
    report progress through the timesteps -- a scripted function can't hold a tqdm
    instance. This adds one Python-level dispatch into compiled code per timestep,
    which is negligible next to the tensor work each step does across the whole
    particle batch.

    The bar is updated exactly ~100 times over the run rather than once per timestep
    (tqdm's own redraw cost would otherwise dominate a 5000-step loop): every
    `steps_count // 100` completed steps triggers one update() call, with any leftover
    remainder from a non-round steps_count flushed once at the end.

    After the loop, logs a deposit-kernel QA line: the largest sigmaL/sigmaT actually
    seen among depositing particles versus the configured kernel radius, so a kernel
    that's silently truncating real physical spread shows up in the log instead of just
    quietly biasing the result.
    """
    num_particles_count = positions_px.size(0)
    device = positions_px.device

    invWidth_per_px = 2.0 / float(width_px - 1)
    invHeight_per_px = 2.0 / float(height_px - 1)
    res_sq_m2 = resolution_m_per_px * resolution_m_per_px

    thermalRetardationScale_dimless = 1.0 / retardationFactor_dimless
    invPorosity_dimless = 1.0 / porosity_frac

    sqrt_2_dt_s05 = math.sqrt(2.0 * timeStep_s)
    # Per-particle energy share; uses the total across all batches so that the sum over
    # ages reproduces the total injected energy exactly. Do NOT additionally divide by
    # steps_count -- the deposits along one path already sweep the full injection profile
    # once, so total deposited == total injected with this weight alone.
    # w = Edot(phase) dt / N_tot
    weight_scale_s = timeStep_s / float(totalSamplesPerSource_count)

    uL_default_dimless = torch.zeros((num_particles_count, 2), device=device)
    uL_default_dimless[:, 0] = 1.0

    # Monotonic latch: once a particle leaves the domain it stays inactive (absorbing
    # outflow boundary -- energy that exits the domain is gone).
    still_in_domain_bool = torch.ones(num_particles_count, dtype=torch.bool, device=device)

    maxSigmaL_px_running = torch.zeros(1, device=device)
    maxSigmaT_px_running = torch.zeros(1, device=device)

    update_every_count = max(1, steps_count // 100)
    pending_count = 0

    with tqdm(total=steps_count, desc=progress_desc, leave=False) as pbar:
        for i_idx in range(steps_count):
            _rwpt_stream_step(
                accumEnergyGrid_flat_J,
                positions_px,
                fieldTensor_m_per_year,
                acceptanceField_1_1_hw,
                sourceIndices_idx,
                injectionEnergyRate_J_per_s,
                birthIndices_count,
                still_in_domain_bool,
                maxSigmaL_px_running,
                maxSigmaT_px_running,
                i_idx,
                steps_count,
                timeStep_s,
                width_px,
                height_px,
                resolution_m_per_px,
                weight_scale_s,
                invWidth_per_px,
                invHeight_per_px,
                res_sq_m2,
                thermalRetardationScale_dimless,
                invPorosity_dimless,
                sqrt_2_dt_s05,
                alphaL_m,
                alphaT_m,
                thermalDiffusivity_m2_per_s,
                secondsPerYear_s,
                uL_default_dimless,
                depositKernelRadius_px,
            )

            pending_count += 1
            if pending_count == update_every_count:
                pbar.update(pending_count)
                pending_count = 0

        if pending_count > 0:
            pbar.update(pending_count)

    maxSigmaL_px_val = float(maxSigmaL_px_running.item())
    maxSigmaT_px_val = float(maxSigmaT_px_running.item())
    widest_sigma_px = max(maxSigmaL_px_val, maxSigmaT_px_val)
    patch_size_px = 2 * depositKernelRadius_px + 1
    log.info(
        f"RWPT[{progress_desc}] deposit kernel QA: max sigmaL={maxSigmaL_px_val:.3f} px, "
        f"max sigmaT={maxSigmaT_px_val:.3f} px, kernel_radius={depositKernelRadius_px} px "
        f"(patch {patch_size_px}x{patch_size_px}, {patch_size_px * resolution_m_per_px:.2f} m across)"
    )
    if widest_sigma_px > 1e-9 and depositKernelRadius_px < 3.0 * widest_sigma_px:
        log.warning(
            f"RWPT[{progress_desc}] deposit kernel radius {depositKernelRadius_px}px covers only "
            f"{depositKernelRadius_px / widest_sigma_px:.1f} sigma of the widest step seen "
            f"(sigmaL={maxSigmaL_px_val:.3f}px, sigmaT={maxSigmaT_px_val:.3f}px); consider raising "
            f"deposit_kernel_radius_px to at least {math.ceil(3.0 * widest_sigma_px)} px so the "
            f"kernel isn't folding a meaningful tail back onto a too-small patch (3 sigma covers "
            f"~99% of the kernel's mass)."
        )


def _orient_field_hw(
    v_m_per_year: torch.Tensor, grid_h_px: int, grid_w_px: int, name: str
) -> torch.Tensor:
    """Return the field in (H, W) layout expected by grid_sample.

    Accepts (W, H) (the caller's native x-major layout) or (H, W). Square grids are
    ambiguous and are treated as (W, H), matching the caller. Anything else is an error
    rather than a silent pass-through.
    """
    if v_m_per_year.shape == (grid_w_px, grid_h_px):
        return v_m_per_year.t().contiguous()
    if v_m_per_year.shape == (grid_h_px, grid_w_px):
        return v_m_per_year
    raise ValueError(
        f"{name} has shape {tuple(v_m_per_year.shape)}; expected "
        f"{(grid_w_px, grid_h_px)} or {(grid_h_px, grid_w_px)}"
    )


def _prepare_velocity_field(
    vx_m_per_year: torch.Tensor, vy_m_per_year: torch.Tensor, grid_h_px: int, grid_w_px: int, device: torch.device
) -> torch.Tensor:
    """Orient the raw vx/vy fields once, at full resolution, into the (1, 2, H, W) shape
    the kernel expects.

    Doing this once up front (rather than inside every generate_physical_plumes call)
    matters for the multi-resolution acceptance driver: a downsampled level can be
    square even when the full grid isn't, which would make _orient_field_hw's (W,H) vs
    (H,W) guess ambiguous again if re-run at every level. Every level instead resamples
    this single, unambiguously-oriented full-resolution field.
    """
    vx_hw_m_per_year = _orient_field_hw(vx_m_per_year.to(device), grid_h_px, grid_w_px, "vx")
    vy_hw_m_per_year = _orient_field_hw(vy_m_per_year.to(device), grid_h_px, grid_w_px, "vy")
    return torch.stack([vx_hw_m_per_year, vy_hw_m_per_year]).unsqueeze(0).float()


def _retardation_factor(modeConstants: Any) -> float:
    """R = (rho c)_aq / (theta rho_w c_w) -- same formula as RwptConfig.retardationFactor_dimless,
    but computable straight from modeConstants before any RwptConfig (and therefore before
    a step count) exists. Needed by _min_steps_for_courant, which has to run BEFORE the
    config that would otherwise supply this property.
    """
    rho_cw_J_per_m3K = modeConstants.water_density_kg_per_m3 * modeConstants.water_specific_heat_J_per_kgK
    rho_cr_J_per_m3K = modeConstants.rock_density_kg_per_m3 * modeConstants.rock_specific_heat_J_per_kgK
    rho_c_aq_J_per_m3K = (
        modeConstants.porosity_frac * rho_cw_J_per_m3K + (1.0 - modeConstants.porosity_frac) * rho_cr_J_per_m3K
    )
    return rho_c_aq_J_per_m3K / (modeConstants.porosity_frac * rho_cw_J_per_m3K)


def _thermal_diffusivity(modeConstants: Any) -> float:
    """D_cond = lambda_wet / (rho c)_aq -- same formula as
    RwptConfig.thermalDiffusivity_m2_per_s, but computable straight from modeConstants
    before any RwptConfig exists. Needed by _auto_deposit_kernel_radius_px, which (like
    _min_steps_for_courant) has to run BEFORE the config it would otherwise read this
    off of.
    """
    rho_cw_J_per_m3K = modeConstants.water_density_kg_per_m3 * modeConstants.water_specific_heat_J_per_kgK
    rho_cr_J_per_m3K = modeConstants.rock_density_kg_per_m3 * modeConstants.rock_specific_heat_J_per_kgK
    rho_c_aq_J_per_m3K = (
        modeConstants.porosity_frac * rho_cw_J_per_m3K + (1.0 - modeConstants.porosity_frac) * rho_cr_J_per_m3K
    )
    return modeConstants.thermal_conductivity_wet_W_per_mK / rho_c_aq_J_per_m3K


def _expected_deposit_sigma_px(
    vField_m_per_year: torch.Tensor,
    resolution_m_per_px: float,
    timeStep_s: float,
    porosity_frac: float,
    retardationFactor_dimless: float,
    alphaL_m: float,
    alphaT_m: float,
    thermalDiffusivity_m2_per_s: float,
) -> float:
    """Worst-case per-step deposit-kernel sigma (px) expected at this resolution/step
    size, from the same dispersion physics local_transport_state evaluates every step:

        D_L/T = (D_cond + alpha_L/T |v_s| / R) / dx^2,  sigma = sqrt(2 D dt)

    evaluated at the field's fastest seepage velocity (mechanical dispersion grows with
    |v_s|, so the fastest-flowing cell anywhere in the field sets the widest kernel any
    particle can actually need) rather than per-particle, since this runs once per level
    to size the kernel radius before any particle exists.
    """
    maxFlux_m_per_year = torch.norm(vField_m_per_year[0], dim=0).max().item()
    maxSeepage_m_per_s = maxFlux_m_per_year / SECONDS_PER_YEAR_S / porosity_frac
    thermalRetardationScale_dimless = 1.0 / retardationFactor_dimless
    res_sq_m2 = resolution_m_per_px * resolution_m_per_px

    diffL_px2_per_s = (
        thermalDiffusivity_m2_per_s + alphaL_m * maxSeepage_m_per_s * thermalRetardationScale_dimless
    ) / res_sq_m2
    diffT_px2_per_s = (
        thermalDiffusivity_m2_per_s + alphaT_m * maxSeepage_m_per_s * thermalRetardationScale_dimless
    ) / res_sq_m2

    sigmaL_px = math.sqrt(max(0.0, 2.0 * diffL_px2_per_s * timeStep_s))
    sigmaT_px = math.sqrt(max(0.0, 2.0 * diffT_px2_per_s * timeStep_s))
    return max(sigmaL_px, sigmaT_px)


def _auto_deposit_kernel_radius_px(
    vField_m_per_year: torch.Tensor,
    resolution_m_per_px: float,
    timeStep_s: float,
    porosity_frac: float,
    retardationFactor_dimless: float,
    alphaL_m: float,
    alphaT_m: float,
    thermalDiffusivity_m2_per_s: float,
    min_radius_px: int,
    max_radius_px: int,
) -> int:
    """Deposit kernel radius (px) sized to cover 3 sigma (~99% of the kernel's mass) of
    the worst-case per-step dispersion at THIS level's own resolution and step size --
    the same idea as _min_steps_for_courant, applied to the deposit patch instead of the
    step count. sigma in pixels scales as 1/dx for a fixed physical dispersion width, so
    this is computed fresh per grid level rather than reused from the fine level.

    Clamped to [min_radius_px, max_radius_px]: max_radius_px is a cost ceiling (the
    yaml-configured deposit_kernel_radius_px, exactly analogous to yaml `steps` capping
    _min_steps_for_courant) so an unusually fast/dispersive field can't blow up the
    (2r+1)^2 per-particle scatter cost unboundedly; min_radius_px keeps the patch from
    degenerating to a single cell when the physics implies almost no spread.
    """
    sigma_px = _expected_deposit_sigma_px(
        vField_m_per_year, resolution_m_per_px, timeStep_s, porosity_frac,
        retardationFactor_dimless, alphaL_m, alphaT_m, thermalDiffusivity_m2_per_s,
    )
    radius_px = math.ceil(3.0 * sigma_px)
    return max(min_radius_px, min(max_radius_px, radius_px))


def _min_steps_for_courant(
    vField_m_per_year: torch.Tensor,
    resolution_m_per_px: float,
    duration_years: float,
    porosity_frac: float,
    retardationFactor_dimless: float,
    max_courant: float,
) -> int:
    """Minimum step count that keeps the thermal Courant number at or below max_courant.

    Co = |v_T|_max * dt / dx, with dt = duration_years * S_yr / steps and
    |v_T|_max = |q|_max / (theta * R) the fastest RETARDED thermal front speed anywhere
    in the field. Requiring Co <= max_courant and solving for steps gives

        steps >= |v_T|_max * duration_years * S_yr / (dx * max_courant)

    i.e. this hands back the smallest step count for which particles don't cross more
    than max_courant cells per step near the fastest-flowing wells -- past that point,
    the source region gets smeared into a disc set by dt rather than by the physics.
    """
    maxFlux_m_per_year = torch.norm(vField_m_per_year[0], dim=0).max().item()
    maxThermal_m_per_s = maxFlux_m_per_year / SECONDS_PER_YEAR_S / (porosity_frac * retardationFactor_dimless)
    required_steps = (
        maxThermal_m_per_s * duration_years * SECONDS_PER_YEAR_S / (resolution_m_per_px * max_courant)
    )
    return max(1, math.ceil(required_steps))


def _log_step_diagnostics(config: RwptConfig, tag: str) -> None:
    """Log the steps/samples actually being used for this solve.

    The step count itself is chosen once, up front in run_rwpt_thermal_prior, via
    _min_steps_for_courant (capped at the yaml-configured max) -- there's nothing left
    to compute or warn about here, just what ended up being used.
    """
    log.info(
        f"RWPT[{tag}] using steps={config.timeSteps_count}, samples={config.samplesPerSource_count} "
        f"(dt={config.timeStep_s / 86400.0:.2f} d, dx={config.resolution_m_per_px:.2f} m)"
    )


def generate_physical_plumes(
    config: RwptConfig,
    heatPumpPositions_px: torch.Tensor,
    vx_m_per_year: torch.Tensor | None,
    vy_m_per_year: torch.Tensor | None,
    dims_px: tuple[int, int],
    acceptance_hw: torch.Tensor | None = None,
    tag: str = "run",
    vField_m_per_year: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return the RAW (unclamped) temperature field in Celsius, shape (H, W).

    acceptance_hw is a per-cell factor in [0, 1] (default all-ones) applied to every
    deposit. Passing the converged acceptance field turns the fixed-flux source into its
    Dirichlet-limited equivalent. The field is not clamped here -- the caller measures
    the raw overshoot to drive the acceptance iteration and clamps only at the very end.

    Either pass raw vx_m_per_year/vy_m_per_year (oriented and stacked internally --
    the original interface, kept for backward compatibility) or a pre-oriented
    vField_m_per_year of shape (1, 2, H, W) (used by the multi-resolution acceptance
    driver, which prepares and resamples one field across several grid levels via
    _prepare_velocity_field and would otherwise repeat the orientation guess -- ambiguous
    for any level that happens to be square -- on every call). If vField_m_per_year is
    given, vx_m_per_year/vy_m_per_year are ignored and may be passed as None.
    """
    device = torch.device(config.device)
    grid_h_px, grid_w_px = int(dims_px[0]), int(dims_px[1])
    numHps_count = len(heatPumpPositions_px)

    with torch.inference_mode():
        if numHps_count == 0:
            return torch.full((grid_h_px, grid_w_px), config.ambientTemp_C, dtype=torch.float32)

        if vField_m_per_year is not None:
            vField_m_per_year = vField_m_per_year.to(device).float()
        else:
            assert vx_m_per_year is not None and vy_m_per_year is not None, (
                "generate_physical_plumes needs either vField_m_per_year or both vx_m_per_year "
                "and vy_m_per_year"
            )
            vField_m_per_year = _prepare_velocity_field(vx_m_per_year, vy_m_per_year, grid_h_px, grid_w_px, device)

        _log_step_diagnostics(config, tag)

        # Acceptance field as (1, 1, H, W); default ones = no suppression.
        if acceptance_hw is None:
            acceptanceField_1_1_hw = torch.ones((1, 1, grid_h_px, grid_w_px), device=device, dtype=torch.float32)
        else:
            acceptanceField_1_1_hw = acceptance_hw.to(device).view(1, 1, grid_h_px, grid_w_px).float()

        # Input positions are (row, col); the kernel works in (x, y) = (col, row).
        positions_rowcol_px = heatPumpPositions_px.to(device)
        positions_xy_px = torch.stack([positions_rowcol_px[:, 1], positions_rowcol_px[:, 0]], dim=1)

        globalEnergy_J = torch.zeros(grid_h_px * grid_w_px, device=device, dtype=torch.float32)

        rho_cw_J_per_m3K = config.waterDensity_kg_per_m3 * config.waterSpecificHeat_J_per_kgK

        # Distribute particle capacity evenly across all sources
        samples_per_hp_batch_count = max(1, config.particles_per_batch_count // numHps_count)
        samples_per_hp_batch_count = min(samples_per_hp_batch_count, config.samplesPerSource_count)
        total_samples_count = config.samplesPerSource_count

        # Edot = Q rho_w c_w (T_inj - T_amb)  [J/s]. This is the only place the injection
        # rate enters: the flow it drives is already in the sampled field.
        injectionEnergyRate_J_per_s = (
            config.injectionRate_m3_per_s * (config.injectionTemp_C - config.ambientTemp_C) * rho_cw_J_per_m3K
        )

        for s_idx in tqdm(
            range(0, total_samples_count, samples_per_hp_batch_count),
            desc=f"RWPT[{tag}] batches",
            leave=False,
        ):
            s_end_idx = min(s_idx + samples_per_hp_batch_count, total_samples_count)
            curr_sub_samples_count = s_end_idx - s_idx
            flat_count = numHps_count * curr_sub_samples_count

            # All particles are released at step 0: under the path->snapshot mapping each
            # path already spans every age 0..T-1, so every age gets the full sample
            # count. Staggering births instead truncates the oldest ages.
            birthIndices_count = torch.zeros(flat_count, dtype=torch.long, device=device)
            srcIndices_idx = torch.arange(numHps_count, device=device).repeat_interleave(curr_sub_samples_count)

            flat_origins_px = positions_xy_px.repeat_interleave(curr_sub_samples_count, dim=0)

            angle_rad = torch.rand(flat_count, device=device) * 2.0 * math.pi
            offset_px = torch.stack([torch.cos(angle_rad), torch.sin(angle_rad)], dim=1) * SEED_RADIUS_PX
            flat_particles_px = flat_origins_px + offset_px

            rwpt_seasonal_stream_kernel(
                globalEnergy_J,
                flat_particles_px,
                vField_m_per_year,
                acceptanceField_1_1_hw,
                srcIndices_idx,
                injectionEnergyRate_J_per_s,
                birthIndices_count,
                float(config.timeStep_s),
                config.timeSteps_count,
                grid_w_px,
                grid_h_px,
                float(config.resolution_m_per_px),
                config.longitudinalDispersivity_m,
                config.transverseDispersivityH_m,
                config.thermalDiffusivity_m2_per_s,
                config.porosity_frac,
                config.retardationFactor_dimless,
                SECONDS_PER_YEAR_S,
                total_samples_count,
                config.depositKernelRadius_px,
                f"RWPT[{tag}] steps (batch {s_idx // samples_per_hp_batch_count + 1})",
            )

        # dT = E / (V_cell (rho c)_aq), V_cell = dx^2 b
        cellVolume_m3 = (config.resolution_m_per_px**2) * config.thicknessAquifer_m
        heatCapacityAquifer_J_per_m3K = config.volumetricHeatCapacityAquifer_J_per_m3K

        energyGrid_2d_J = globalEnergy_J.view(grid_h_px, grid_w_px)
        deltaTempMap_C = energyGrid_2d_J / (cellVolume_m3 * heatCapacityAquifer_J_per_m3K)
        finalTempMap_C = config.ambientTemp_C + deltaTempMap_C

        return finalTempMap_C.cpu()


# --------------------------------------------------------------------------------- #
# Acceptance-field iteration (multi-resolution)
# --------------------------------------------------------------------------------- #

def _gaussian_blur_hw(img_hw: torch.Tensor, sigma_px: float) -> torch.Tensor:
    """Separable Gaussian blur on an (H, W) map with replicate padding (same size out)."""
    if sigma_px <= 0.0:
        return img_hw
    radius = max(1, int(round(3.0 * sigma_px)))
    xs = torch.arange(-radius, radius + 1, dtype=torch.float32, device=img_hw.device)
    k = torch.exp(-0.5 * (xs / sigma_px) ** 2)
    k = (k / k.sum()).to(img_hw.dtype)
    img = img_hw.view(1, 1, img_hw.size(0), img_hw.size(1))
    img = F.pad(img, (radius, radius, radius, radius), mode="replicate")
    img = F.conv2d(img, k.view(1, 1, 1, -1))
    img = F.conv2d(img, k.view(1, 1, -1, 1))
    return img.view(img_hw.size(0), img_hw.size(1))


@dataclass
class _RwptVizSpec:
    """Destination and physical-band metadata for per-stage diagnostic PNGs."""

    out_dir: Path
    ambient_temp_C: float
    min_temp_C: float
    max_temp_C: float
    temp_spread_C: float


def _save_rwpt_viz(
    viz: _RwptVizSpec | None,
    stem: str,
    title: str,
    *,
    resolution_m_per_px: float,
    wells_rowcol_px: torch.Tensor | None = None,
    temp_C: torch.Tensor | None = None,
    acceptance: torch.Tensor | None = None,
    acceptance_next: torch.Tensor | None = None,
    velocity_m_per_year: torch.Tensor | None = None,
) -> None:
    if viz is None:
        return
    visualize_rwpt_fields(
        viz.out_dir / stem,
        title,
        ambient_temp_C=viz.ambient_temp_C,
        min_temp_C=viz.min_temp_C,
        max_temp_C=viz.max_temp_C,
        temp_spread_C=viz.temp_spread_C,
        resolution_m_per_px=resolution_m_per_px,
        temp_C=temp_C,
        acceptance=acceptance,
        acceptance_next=acceptance_next,
        velocity_m_per_year=velocity_m_per_year,
        wells_rowcol_px=wells_rowcol_px,
    )


def _overshoot_beyond_band_C(tempMap_C: torch.Tensor, min_temp_C: float, max_temp_C: float) -> float:
    """Worst temperature (in C) by which any cell exceeds the physical band."""
    hi = (tempMap_C - max_temp_C).clamp(min=0.0).max().item()
    lo = (min_temp_C - tempMap_C).clamp(min=0.0).max().item()
    return max(hi, lo)


def _argmax_overshoot_cell(tempMap_C: torch.Tensor, min_temp_C: float, max_temp_C: float) -> tuple[int, int]:
    """Row/col of the single worst-overshoot cell.

    Logged alongside the overshoot magnitude so it's visible whether the acceptance
    iteration is grinding down the SAME hot spot each pass (a slow/weak correction) or
    chasing a DIFFERENT well's hot spot every iteration ("whack-a-mole" across many
    wells competing for the single global worst-cell statistic).
    """
    hi_C = (tempMap_C - max_temp_C).clamp(min=0.0)
    lo_C = (min_temp_C - tempMap_C).clamp(min=0.0)
    worst_C = torch.maximum(hi_C, lo_C)
    flat_idx = int(torch.argmax(worst_C).item())
    row_idx = flat_idx // tempMap_C.size(1)
    col_idx = flat_idx % tempMap_C.size(1)
    return row_idx, col_idx


def _update_acceptance(
    acceptance_hw: torch.Tensor,
    tempMap_C: torch.Tensor,
    ambient_temp_C: float,
    min_temp_C: float,
    max_temp_C: float,
    relaxation: float,
    blur_sigma_px: float,
) -> torch.Tensor:
    """One self-consistent update of the per-cell acceptance field.

    Deviation from ambient is (approximately) linear in deposited energy, which is linear
    in acceptance:  dev(x) ~= dev_full(x) * a(x).  The sustainable deviation ("cap") is
    (T_inj - T_amb) on the warm side and (T_amb - T_inj) on the cold side, i.e. the band
    bound relative to ambient. Solving dev = cap for a gives the target
        a*(x) = a(x) * cap / dev(x),
    which under the linear model equals cap / dev_full(x) regardless of the current a.
    We under-relax toward a* (relaxation < 1) to damp the upstream->downstream coupling
    that the linear model ignores, then blur so the boundary isn't ragged. Cells whose
    side has no cap (e.g. sub-ambient noise under warm-only injection) are left
    untouched; the final clamp handles that residue.
    """
    dev_C = tempMap_C - ambient_temp_C
    cap_hi_C = max_temp_C - ambient_temp_C   # >= 0
    cap_lo_C = min_temp_C - ambient_temp_C   # <= 0

    # Signed cap on the side each cell deviates toward.
    cap_C = torch.where(
        dev_C >= 0.0,
        torch.full_like(dev_C, cap_hi_C),
        torch.full_like(dev_C, cap_lo_C),
    )
    has_cap = cap_C.abs() > 1e-9
    ratio = torch.where(
        has_cap, dev_C / torch.where(has_cap, cap_C, torch.ones_like(cap_C)), torch.zeros_like(dev_C)
    )

    # correction = cap/dev = 1/ratio: <1 suppresses overshoot, >1 recovers over-
    # suppression (capped by the [0,1] clamp on the suggested acceptance). ratio~0 -> hold.
    correction = torch.where(ratio > 1e-6, 1.0 / ratio, torch.ones_like(ratio))
    suggested_hw = (acceptance_hw * correction).clamp(0.0, 1.0)

    new_acc_hw = acceptance_hw + relaxation * (suggested_hw - acceptance_hw)
    new_acc_hw = _gaussian_blur_hw(new_acc_hw, blur_sigma_px).clamp(0.0, 1.0)
    return new_acc_hw


def _run_acceptance_stage(
    config: RwptConfig,
    heatPumpPositions_px: torch.Tensor,
    vField_m_per_year: torch.Tensor,
    dims_px: tuple[int, int],
    acceptance_hw: torch.Tensor,
    max_iters: int,
    threshold_C: float,
    relaxation: float,
    blur_sigma_px: float,
    ambient_temp_C: float,
    min_temp_C: float,
    max_temp_C: float,
    tag: str,
    viz: _RwptVizSpec | None = None,
    viz_stem: str = "",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Iterate solve -> measure overshoot -> update acceptance until converged or capped.

    Returns (last raw tempMap_C, converged acceptance_hw). The tempMap returned is the
    one from the solve that met the threshold (or the last solve if max_iters is hit),
    i.e. it already reflects the acceptance in force when it was produced.

    vField_m_per_year must already be at this stage's own resolution (see
    _prepare_velocity_field / the multi-resolution driver in run_rwpt_thermal_prior).
    """
    tempMap_C = torch.full((int(dims_px[0]), int(dims_px[1])), ambient_temp_C, dtype=torch.float32)
    overshoot_C = float("inf")
    for it in range(max_iters):
        acc_used_hw = acceptance_hw
        tempMap_C = generate_physical_plumes(
            config, heatPumpPositions_px, None, None, dims_px,
            acceptance_hw=acceptance_hw, tag=f"{tag} it{it}", vField_m_per_year=vField_m_per_year,
        )
        overshoot_C = _overshoot_beyond_band_C(tempMap_C, min_temp_C, max_temp_C)
        worst_row_idx, worst_col_idx = _argmax_overshoot_cell(tempMap_C, min_temp_C, max_temp_C)
        log.info(
            f"RWPT[{tag}] iter {it}: max overshoot beyond band = {overshoot_C:.4f} C "
            f"(threshold {threshold_C:.4f} C) at cell (row={worst_row_idx}, col={worst_col_idx}), "
            f"acceptance min={acceptance_hw.min():.3f}"
        )
        converged = overshoot_C < threshold_C
        acc_next_hw = None
        if not converged:
            acceptance_hw = _update_acceptance(
                acceptance_hw, tempMap_C, ambient_temp_C, min_temp_C, max_temp_C, relaxation, blur_sigma_px
            )
            acc_next_hw = acceptance_hw
        _save_rwpt_viz(
            viz,
            f"{viz_stem}_it{it:02d}",
            (
                f"RWPT[{tag}] iter {it}: overshoot {overshoot_C:.3f} C "
                f"at (row={worst_row_idx}, col={worst_col_idx}), "
                f"acceptance min={float(acc_used_hw.min()):.3f}"
            ),
            resolution_m_per_px=config.resolution_m_per_px,
            wells_rowcol_px=heatPumpPositions_px,
            temp_C=tempMap_C,
            acceptance=acc_used_hw,
            acceptance_next=acc_next_hw,
        )
        if converged:
            log.info(f"RWPT[{tag}] converged at iter {it} (overshoot {overshoot_C:.4f} C < {threshold_C:.4f} C)")
            break
    else:
        log.warning(
            f"RWPT[{tag}] hit max_iters={max_iters} with overshoot {overshoot_C:.4f} C "
            f">= threshold {threshold_C:.4f} C; next grid level (or the final clamp) will "
            f"have to absorb the residual."
        )
    return tempMap_C, acceptance_hw


def _build_grid_levels(grid_h_px: int, grid_w_px: int, fractions: list[float]) -> list[tuple[int, int]]:
    """Turn a list of resolution fractions into (h, w) pixel sizes.

    Each dimension is floored to at least 8px so a coarse level still has enough cells
    for a meaningful acceptance field. The last level is forced to exactly
    (grid_h_px, grid_w_px) regardless of rounding, since that's the resolution the
    final output must actually be returned at.
    """
    assert len(fractions) >= 1, "acceptance_grid_fractions must have at least one entry"
    levels: list[tuple[int, int]] = []
    for frac in fractions:
        level_h = max(8, int(round(grid_h_px * frac)))
        level_w = max(8, int(round(grid_w_px * frac)))
        levels.append((level_h, level_w))
    levels[-1] = (grid_h_px, grid_w_px)
    return levels


def _scale_positions_to_level(
    heatPumpPositions_px: torch.Tensor, full_dims_px: tuple[int, int], level_dims_px: tuple[int, int]
) -> torch.Tensor:
    """Rescale (row, col) well positions from the full-resolution grid to a level's grid."""
    full_h, full_w = full_dims_px
    level_h, level_w = level_dims_px
    scaled_px = heatPumpPositions_px.clone().float()
    scaled_px[:, 0] = scaled_px[:, 0] * (float(level_h) / float(full_h))
    scaled_px[:, 1] = scaled_px[:, 1] * (float(level_w) / float(full_w))
    return scaled_px


def _build_config(
    modeConstants: Any,
    numHps_count: int,
    steps: int,
    samples: int,
    particles_per_batch: int,
    device: torch.device,
    duration_years: float,
    deposit_kernel_radius_px: int,
    resolution_m_per_px: float,
) -> tuple[RwptConfig, int]:
    """Build an RwptConfig with the injection profile resampled to `steps` time steps,
    at the given `resolution_m_per_px` (which may differ from modeConstants.resolution_m
    for a coarse acceptance-grid level -- see run_rwpt_thermal_prior)."""
    rate_m3_per_s, cyc1 = convert_injection_config(
        modeConstants.injection_rate_m3_per_s, steps, duration_years, device
    )
    temp_C, cyc2 = convert_injection_config(
        modeConstants.injection_temperature_C, steps, duration_years, device
    )
    assert cyc1 == cyc2, "Seasonal cycle step mismatch between rate and temperature"

    cfg = RwptConfig(
        device=str(device),
        samplesPerSource_count=samples,
        timeSteps_count=steps,
        resolution_m_per_px=resolution_m_per_px,
        ambientTemp_C=modeConstants.ambient_temperature_C,
        injectionRate_m3_per_s=rate_m3_per_s.unsqueeze(0).repeat(numHps_count, 1),
        injectionTemp_C=temp_C.unsqueeze(0).repeat(numHps_count, 1),
        timeEnd_years=duration_years,
        seasonalCycleSteps_count=cyc1,
        porosity_frac=modeConstants.porosity_frac,
        rockDensity_kg_per_m3=modeConstants.rock_density_kg_per_m3,
        rockSpecificHeat_J_per_kgK=modeConstants.rock_specific_heat_J_per_kgK,
        waterDensity_kg_per_m3=modeConstants.water_density_kg_per_m3,
        waterSpecificHeat_J_per_kgK=modeConstants.water_specific_heat_J_per_kgK,
        thermalConductivityDry_W_per_mK=modeConstants.thermal_conductivity_dry_W_per_mK,
        thermalConductivityWet_W_per_mK=modeConstants.thermal_conductivity_wet_W_per_mK,
        thicknessAquifer_m=modeConstants.thickness_aquifer_m,
        longitudinalDispersivity_m=modeConstants.longitudinal_dispersivity_m,
        transverseDispersivityH_m=modeConstants.transverse_dispersivity_h_m,
        particles_per_batch_count=particles_per_batch,
        depositKernelRadius_px=deposit_kernel_radius_px,
    )
    return cfg, cyc1


def run_rwpt_thermal_prior(
    mode_rwpt: Any,
    modeConstants: Any,
    heatPumpPositions_px: torch.Tensor,
    vx_m_per_year: torch.Tensor,
    vy_m_per_year: torch.Tensor,
    dims_px: tuple[int, int],
    viz_dir: Path | None = None,
) -> torch.Tensor:
    """Multi-resolution self-consistent thermal prior, returned as a single [0, 1] map
    (H, W).

    The acceptance field is solved on a sequence of increasingly fine spatial grids
    (see the module docstring for why), each level warm-started by bilinearly upsampling
    the previous level's converged acceptance field. Only the final (full-resolution)
    level's temperature map is used for the output; earlier levels exist purely to find
    a good low-frequency acceptance pattern cheaply before refining it. A light final
    clamp removes any residual Monte-Carlo overshoot before normalization.

    Step counts are chosen automatically per level from the thermal Courant number
    (see _min_steps_for_courant), each capped at that level's yaml-configured step count
    (`steps` for the final level, `coarse_steps` for every other level) -- the yaml value
    is a cost ceiling that is never exceeded, but is used as-is only when the flow
    actually needs that many steps; a slower flow (or a coarser, physically-larger-pixel
    level) uses fewer, cheaper steps for the same Courant target.

    Tunable via mode_rwpt attributes (defaults in parentheses):
        samples                       -- fine (final-level) sample count
        steps                         -- fine (final-level) time steps, as a MAX (see above)
        target_courant                (1.0)         -- Courant number the step-count
                                                        selection targets
        coarse_samples                (10_000)      -- sample count for every non-final level
        coarse_steps                  (= steps)      -- time steps for every non-final level, as a MAX
        acceptance_grid_fractions     ([0.1, 0.3, 1.0])  -- grid size per level, as a
                                                             fraction of the full grid
        acceptance_grid_max_iters     ([4, 4, 3])    -- max iterations per level (last
                                                          value repeats if shorter than
                                                          acceptance_grid_fractions)
        convergence_threshold_C       (0.05)
        acceptance_relaxation         (0.7)
        acceptance_blur_px            (1.0)
        particles_per_batch           (5_000_000)   -- for the final level; non-final
                                                        levels run in a single batch sized
                                                        to fit all their (few) samples
        deposit_kernel_radius_px      (8)           -- MAX deposit-kernel radius (px);
                                                        each level's actual radius is
                                                        computed automatically from that
                                                        level's own dispersion physics
                                                        (see _auto_deposit_kernel_radius_px)
                                                        and capped at this value, same
                                                        pattern as `steps` capping the
                                                        Courant-derived step count
        min_deposit_kernel_radius_px  (2)            -- floor on the auto-computed radius
        auto_deposit_kernel_radius    (True)         -- set False to skip the automatic
                                                        sizing and just use
                                                        deposit_kernel_radius_px as a
                                                        fixed radius at every level

    If viz_dir is set, a PNG is written at the start of each grid level (coarse |q| +
    warm-started acceptance) and after every acceptance iteration (temp prior,
    acceptance, overshoot), plus the post-clamp field. Tests omit viz_dir and skip I/O.
    """
    device = heatPumpPositions_px.device
    numHps_count = len(heatPumpPositions_px)
    duration_years = modeConstants.duration_years

    ambient_temp_C = modeConstants.ambient_temperature_C
    temp_spread_C = modeConstants.temperature_spread_C

    # Iteration controls
    coarse_samples = int(getattr(mode_rwpt, "coarse_samples", DEFAULT_COARSE_SAMPLES))
    coarse_steps = int(getattr(mode_rwpt, "coarse_steps", mode_rwpt.steps))
    threshold_C = float(getattr(mode_rwpt, "convergence_threshold_C", DEFAULT_CONVERGENCE_THRESHOLD_C))
    relaxation = float(getattr(mode_rwpt, "acceptance_relaxation", DEFAULT_ACCEPTANCE_RELAXATION))
    blur_px = float(getattr(mode_rwpt, "acceptance_blur_px", DEFAULT_ACCEPTANCE_BLUR_PX))
    fine_ppb = int(getattr(mode_rwpt, "particles_per_batch", 2_000_000))
    max_deposit_kernel_radius_px = int(
        getattr(mode_rwpt, "deposit_kernel_radius_px", DEFAULT_DEPOSIT_KERNEL_RADIUS_PX)
    )
    min_deposit_kernel_radius_px = int(
        getattr(mode_rwpt, "min_deposit_kernel_radius_px", DEFAULT_MIN_DEPOSIT_KERNEL_RADIUS_PX)
    )
    auto_deposit_kernel_radius = bool(getattr(mode_rwpt, "auto_deposit_kernel_radius", False))

    acceptance_grid_fractions = list(
        getattr(mode_rwpt, "acceptance_grid_fractions", DEFAULT_ACCEPTANCE_GRID_FRACTIONS)
    )
    acceptance_grid_max_iters = list(
        getattr(mode_rwpt, "acceptance_grid_max_iters", DEFAULT_ACCEPTANCE_GRID_MAX_ITERS)
    )
    if len(acceptance_grid_max_iters) < len(acceptance_grid_fractions):
        acceptance_grid_max_iters = acceptance_grid_max_iters + [acceptance_grid_max_iters[-1]] * (
            len(acceptance_grid_fractions) - len(acceptance_grid_max_iters)
        )

    target_courant = float(getattr(mode_rwpt, "target_courant", DEFAULT_TARGET_COURANT))

    grid_h_px, grid_w_px = int(dims_px[0]), int(dims_px[1])

    # Prepared once so both the fine-level step calculation below and the per-level
    # resampling in the acceptance loop reuse the same unambiguously-oriented field.
    vField_full_m_per_year = _prepare_velocity_field(vx_m_per_year, vy_m_per_year, grid_h_px, grid_w_px, device)
    retardation_dimless = _retardation_factor(modeConstants)
    thermal_diffusivity_m2_per_s = _thermal_diffusivity(modeConstants)

    def _resolve_kernel_radius_px(vField_level_m_per_year: torch.Tensor, level_resolution_m: float, level_dt_s: float) -> int:
        if not auto_deposit_kernel_radius:
            return max_deposit_kernel_radius_px
        return _auto_deposit_kernel_radius_px(
            vField_level_m_per_year, level_resolution_m, level_dt_s,
            modeConstants.porosity_frac, retardation_dimless,
            modeConstants.longitudinal_dispersivity_m, modeConstants.transverse_dispersivity_h_m,
            thermal_diffusivity_m2_per_s, min_deposit_kernel_radius_px, max_deposit_kernel_radius_px,
        )

    # Steps needed to keep the thermal Courant number at target_courant at full
    # resolution, capped at whatever the yaml configured as `steps` -- the yaml value is
    # a cost ceiling, never a floor: if the flow is slow enough that fewer steps already
    # satisfy the Courant condition, use fewer (cheaper for the same quality); if the
    # flow needs more than the yaml allows, cap at the yaml value rather than silently
    # blowing past the configured cost budget.
    required_fine_steps = _min_steps_for_courant(
        vField_full_m_per_year, modeConstants.resolution_m, duration_years,
        modeConstants.porosity_frac, retardation_dimless, target_courant,
    )
    fine_steps_used = min(int(mode_rwpt.steps), required_fine_steps)
    log.info(
        f"RWPT fine steps: {required_fine_steps} needed for Courant<={target_courant:.2f} at full "
        f"resolution (dx={modeConstants.resolution_m:.2f} m), yaml max={mode_rwpt.steps} -> using {fine_steps_used}"
    )

    # Fine (full-resolution) config, built once: it both solves the final level and
    # supplies the physical band bounds (injection temperature extremes) used by every
    # level's overshoot check and the final clamp.
    fine_time_step_s = (duration_years * SECONDS_PER_YEAR_S) / float(fine_steps_used)
    fine_kernel_radius_px = _resolve_kernel_radius_px(
        vField_full_m_per_year, modeConstants.resolution_m, fine_time_step_s
    )
    log.info(
        f"RWPT fine deposit kernel radius: {fine_kernel_radius_px} px "
        f"(max={max_deposit_kernel_radius_px} px, auto={auto_deposit_kernel_radius})"
    )
    fine_cfg, _ = _build_config(
        modeConstants, numHps_count, fine_steps_used, mode_rwpt.samples, fine_ppb, device, duration_years,
        fine_kernel_radius_px, modeConstants.resolution_m,
    )
    inj_min_C = float(fine_cfg.injectionTemp_C.min().item())
    inj_max_C = float(fine_cfg.injectionTemp_C.max().item())
    min_temp_C = min(inj_min_C, ambient_temp_C)
    max_temp_C = max(inj_max_C, ambient_temp_C)
    viz = None if viz_dir is None else _RwptVizSpec(
        out_dir=Path(viz_dir),
        ambient_temp_C=ambient_temp_C,
        min_temp_C=min_temp_C,
        max_temp_C=max_temp_C,
        temp_spread_C=temp_spread_C,
    )

    if numHps_count == 0:
        tempMap_C = torch.full((grid_h_px, grid_w_px), ambient_temp_C, dtype=torch.float32)
    else:
        grid_levels_px = _build_grid_levels(grid_h_px, grid_w_px, acceptance_grid_fractions)

        level0_h, level0_w = grid_levels_px[0]
        acceptance_hw = torch.ones((level0_h, level0_w), dtype=torch.float32)
        tempMap_C = torch.full((grid_h_px, grid_w_px), ambient_temp_C, dtype=torch.float32)

        for level_idx, (level_h, level_w) in enumerate(grid_levels_px):
            is_final_level = level_idx == len(grid_levels_px) - 1
            level_vField_m_per_year = F.interpolate(
                vField_full_m_per_year, size=(level_h, level_w), mode="bilinear", align_corners=True
            )
            # Same physical domain, fewer/more pixels -> proportionally larger/smaller
            # physical pixel size. Averaging the h/w scale keeps it sane for a level
            # whose rounded aspect ratio drifted slightly from the full grid's.
            level_scale = 0.5 * (grid_h_px / level_h + grid_w_px / level_w)
            level_resolution_m_per_px = modeConstants.resolution_m * level_scale
            level_positions_px = _scale_positions_to_level(
                heatPumpPositions_px, (grid_h_px, grid_w_px), (level_h, level_w)
            ).to(device)

            if is_final_level:
                level_cfg = fine_cfg
            else:
                # Same Courant-based selection as the fine level, but at this level's
                # own (coarser) resolution and capped at coarse_steps rather than
                # mode_rwpt.steps: a coarser spatial grid has physically larger pixels,
                # so it typically needs far fewer steps for the same Courant target.
                required_level_steps = _min_steps_for_courant(
                    level_vField_m_per_year, level_resolution_m_per_px, duration_years,
                    modeConstants.porosity_frac, retardation_dimless, target_courant,
                )
                level_steps_used = min(coarse_steps, required_level_steps)
                level_ppb = max(1, numHps_count) * coarse_samples
                level_time_step_s = (duration_years * SECONDS_PER_YEAR_S) / float(level_steps_used)
                level_kernel_radius_px = _resolve_kernel_radius_px(
                    level_vField_m_per_year, level_resolution_m_per_px, level_time_step_s
                )
                level_cfg, _ = _build_config(
                    modeConstants, numHps_count, level_steps_used, coarse_samples, level_ppb, device,
                    duration_years, level_kernel_radius_px, level_resolution_m_per_px,
                )

            log.info(
                f"RWPT acceptance level {level_idx}/{len(grid_levels_px) - 1}: "
                f"{level_h}x{level_w} px (dx={level_resolution_m_per_px:.2f} m), "
                f"{level_cfg.samplesPerSource_count} samples/source, "
                f"{level_cfg.timeSteps_count} steps, deposit_kernel_radius={level_cfg.depositKernelRadius_px} px "
                f"(max={max_deposit_kernel_radius_px} px), max_iters={acceptance_grid_max_iters[level_idx]}"
            )

            viz_stem = f"L{level_idx:02d}_{level_h}x{level_w}"
            _save_rwpt_viz(
                viz,
                f"{viz_stem}_grid",
                (
                    f"RWPT level {level_idx}/{len(grid_levels_px) - 1}: "
                    f"coarse grid {level_h}x{level_w} px, warm-start acceptance "
                    f"min={float(acceptance_hw.min()):.3f}"
                ),
                resolution_m_per_px=level_resolution_m_per_px,
                wells_rowcol_px=level_positions_px,
                acceptance=acceptance_hw,
                velocity_m_per_year=level_vField_m_per_year,
            )

            tempMap_C, acceptance_hw = _run_acceptance_stage(
                level_cfg, level_positions_px, level_vField_m_per_year, (level_h, level_w),
                acceptance_hw, acceptance_grid_max_iters[level_idx], threshold_C, relaxation, blur_px,
                ambient_temp_C, min_temp_C, max_temp_C, tag=f"L{level_idx}({level_h}x{level_w})",
                viz=viz, viz_stem=viz_stem,
            )

            if not is_final_level:
                next_h, next_w = grid_levels_px[level_idx + 1]
                acceptance_hw = F.interpolate(
                    acceptance_hw.view(1, 1, level_h, level_w), size=(next_h, next_w),
                    mode="bilinear", align_corners=True,
                ).view(next_h, next_w).clamp(0.0, 1.0)
                _save_rwpt_viz(
                    viz,
                    f"{viz_stem}_upsampled",
                    (
                        f"RWPT level {level_idx} acceptance upsampled to "
                        f"{next_h}x{next_w} px (min={float(acceptance_hw.min()):.3f})"
                    ),
                    resolution_m_per_px=modeConstants.resolution_m * 0.5 * (
                        grid_h_px / next_h + grid_w_px / next_w
                    ),
                    wells_rowcol_px=_scale_positions_to_level(
                        heatPumpPositions_px, (grid_h_px, grid_w_px), (next_h, next_w)
                    ).to(device),
                    acceptance=acceptance_hw,
                    temp_C=F.interpolate(
                        tempMap_C.view(1, 1, level_h, level_w), size=(next_h, next_w),
                        mode="bilinear", align_corners=True,
                    ).view(next_h, next_w),
                )

    # --- Final clamp: absorb residual Monte-Carlo overshoot into the physical band ---
    residual_C = _overshoot_beyond_band_C(tempMap_C, min_temp_C, max_temp_C)
    log.info(
        f"RWPT final clamp: residual overshoot {residual_C:.4f} C into band "
        f"[{min_temp_C:.3f}, {max_temp_C:.3f}] C"
    )
    tempMap_C = torch.clamp(tempMap_C, min=min_temp_C, max=max_temp_C)
    _save_rwpt_viz(
        viz,
        "final_clamped",
        f"RWPT final clamped temperature (residual overshoot was {residual_C:.3f} C)",
        resolution_m_per_px=modeConstants.resolution_m,
        wells_rowcol_px=heatPumpPositions_px,
        temp_C=tempMap_C,
        acceptance=acceptance_hw if numHps_count > 0 else None,
    )

    # --- Normalize to [0, 1] for the CNN (three-branch, matching the injection sign) ---
    if ambient_temp_C <= tempMap_C.min().item():
        normalizedMap_dimless = (tempMap_C - ambient_temp_C) / temp_spread_C
    elif tempMap_C.max().item() <= ambient_temp_C:
        normalizedMap_dimless = (tempMap_C - (ambient_temp_C - temp_spread_C)) / temp_spread_C
    else:
        normalizedMap_dimless = (tempMap_C - (ambient_temp_C - temp_spread_C)) / (2.0 * temp_spread_C)

    log.info(
        f"RWPT thermal prior: min={normalizedMap_dimless.min():.3f}, "
        f"max={normalizedMap_dimless.max():.3f}, mean={normalizedMap_dimless.mean():.4f}"
    )
    return torch.clamp(normalizedMap_dimless, min=0.0, max=1.0)


# --- PHYSICAL VALIDATION AND UNIT TESTS ---


def verify_rwpt_physical_validity(device: str = "cpu") -> None:
    """
    Validates physical correctness against analytical benchmarks:
    1. Pure 2D isotropic Brownian diffusion MSD: <r^2> = 4 * D * t.
    2. Pure uniform advective displacement: <x> = v * t.
    3. Bilinear Cloud-in-Cell interpolation mass conservation.
    4. Absorbing-boundary latch (exited particles stay inactive).
    5. Kinematic seasonal phase (T-1-age)%cycle — travel-time stripes, not i%cycle cancel.
    6. Retarded dispersion coefficients carry no spurious 1/porosity factor.
    7. Plume half-width in a uniform field matches sqrt(2 * alpha_T * L).
    """
    dev = torch.device(device)
    tolerance_rel_frac = 0.05

    dt_s = 100.0
    steps_count = 200
    n_particles_count = 50000
    d_physical_m2_per_s = 1.5e-6
    resolution_m_per_px = 1.0

    d_px2_per_s = d_physical_m2_per_s / (resolution_m_per_px**2)
    sigma_px = math.sqrt(2.0 * d_px2_per_s * dt_s)

    pos_px = torch.zeros((n_particles_count, 2), device=dev, dtype=torch.float32)
    for _ in range(steps_count):
        pos_px += torch.randn((n_particles_count, 2), device=dev) * sigma_px

    msd_empirical_px2 = torch.mean(torch.sum(pos_px**2, dim=1)).item()
    t_total_s = dt_s * steps_count
    msd_analytical_px2 = 4.0 * d_px2_per_s * t_total_s

    diff_err_frac = abs(msd_empirical_px2 - msd_analytical_px2) / msd_analytical_px2
    assert diff_err_frac < tolerance_rel_frac, (
        f"MSD test failed: Empirical={msd_empirical_px2:.4f}, Analytical={msd_analytical_px2:.4f}, Error={diff_err_frac:.4f}"
    )

    v_const_px_per_s = torch.tensor([1.25, -0.75], device=dev)
    pos_adv_px = torch.zeros((n_particles_count, 2), device=dev)
    for _ in range(steps_count):
        pos_adv_px += v_const_px_per_s * dt_s

    mean_disp_empirical_px = torch.mean(pos_adv_px, dim=0)
    disp_analytical_px = v_const_px_per_s * t_total_s
    adv_err_frac = torch.norm(mean_disp_empirical_px - disp_analytical_px) / torch.norm(disp_analytical_px)
    assert adv_err_frac.item() < 1e-5, f"Advection drift test failed: Error={adv_err_frac.item():.6e}"

    grid_w_px, grid_h_px = 50, 50
    grid_test_J = torch.zeros(grid_w_px * grid_h_px, device=dev, dtype=torch.float32)
    rand_pos_px = torch.rand((1000, 2), device=dev) * 40.0 + 5.0
    rand_weights_J = torch.rand(1000, device=dev) * 100.0
    active_bool = torch.ones(1000, dtype=torch.bool, device=dev)

    deposit_energy_cic(grid_test_J, rand_pos_px, rand_weights_J, active_bool, grid_w_px, grid_h_px)
    deposited_sum_J = torch.sum(grid_test_J).item()
    expected_sum_J = torch.sum(rand_weights_J).item()

    cic_err_frac = abs(deposited_sum_J - expected_sum_J) / expected_sum_J
    assert cic_err_frac < 1e-5, f"CIC mass conservation failed: Dep={deposited_sum_J:.4f}, Exp={expected_sum_J:.4f}"

    # 4. Absorbing boundary latch — particle near edge with outward flux stays out
    h_px = w_px = 32
    steps_k_count = 30
    grid_J = torch.zeros(h_px * w_px, device=dev, dtype=torch.float32)
    vx_m_per_year = torch.full((h_px, w_px), 5.0e5, device=dev, dtype=torch.float32)
    vy_m_per_year = torch.zeros((h_px, w_px), device=dev, dtype=torch.float32)
    field_m_per_year = torch.stack([vx_m_per_year, vy_m_per_year]).unsqueeze(0)
    acceptance_k_hw = torch.ones((1, 1, h_px, w_px), device=dev, dtype=torch.float32)
    pos_k_px = torch.tensor([[float(w_px - 2), float(h_px // 2)]], device=dev)
    energy_J_per_s = torch.zeros((1, steps_k_count), device=dev)
    birth_count = torch.zeros(1, dtype=torch.long, device=dev)
    src_idx = torch.zeros(1, dtype=torch.long, device=dev)
    rwpt_seasonal_stream_kernel(
        grid_J,
        pos_k_px,
        field_m_per_year,
        acceptance_k_hw,
        src_idx,
        energy_J_per_s,
        birth_count,
        86400.0 * 30.0,
        steps_k_count,
        w_px,
        h_px,
        5.0,
        0.0,
        0.0,
        0.0,
        0.25,
        5.0,
        SECONDS_PER_YEAR_S,
        1,
    )
    still_inside_bool = bool(
        (pos_k_px[0, 0] >= 0.0)
        and (pos_k_px[0, 0] <= float(w_px - 1))
        and (pos_k_px[0, 1] >= 0.0)
        and (pos_k_px[0, 1] <= float(h_px - 1))
    )
    assert not still_inside_bool, (
        f"Boundary latch test: particle expected outside domain, got pos={pos_k_px.tolist()}"
    )

    # 5. Kinematic seasonal phase: phase = (T-1-age)%C depends on birth mid-path
    n_samp_count = 4
    steps5_count = 8
    dt5_s = 1.0
    e_rates_J_per_s = torch.arange(1.0, steps5_count + 1.0, device=dev).unsqueeze(0)
    grid5_J = torch.zeros(h_px * w_px, device=dev)
    field0_m_per_year = torch.zeros((1, 2, h_px, w_px), device=dev)
    acceptance5_hw = torch.ones((1, 1, h_px, w_px), device=dev, dtype=torch.float32)
    births_count = torch.arange(n_samp_count, device=dev, dtype=torch.long)
    srcs_idx = torch.zeros(n_samp_count, dtype=torch.long, device=dev)
    pos5_px = torch.full((n_samp_count, 2), float(w_px // 2), device=dev)
    rwpt_seasonal_stream_kernel(
        grid5_J,
        pos5_px,
        field0_m_per_year,
        acceptance5_hw,
        srcs_idx,
        e_rates_J_per_s,
        births_count,
        dt5_s,
        steps5_count,
        w_px,
        h_px,
        5.0,
        0.0,
        0.0,
        0.0,
        0.25,
        5.0,
        SECONDS_PER_YEAR_S,
        n_samp_count,
    )
    deposited_J = float(grid5_J.sum().item())
    expected_J = 0.0
    for i_idx in range(steps5_count):
        for b_idx in range(n_samp_count):
            if i_idx >= b_idx:
                inj_idx = steps5_count - 1 - (i_idx - b_idx)
                expected_J += e_rates_J_per_s[0, inj_idx].item() * dt5_s / n_samp_count
    rel_frac = abs(deposited_J - expected_J) / max(expected_J, 1e-12)
    assert rel_frac < 1e-4, f"Kinematic season energy failed: dep={deposited_J:.6f} exp={expected_J:.6f} rel={rel_frac:.2e}"

    # 6. Dispersion scaling: D_L must equal D_cond + alpha_L |v_s| / R exactly.
    # A uniform Darcy flux q gives |v_s| = q / theta, so any stray 1/theta shows up here.
    porosity_frac = 0.25
    retardation_dimless = 2.0
    alphaL_m = 3.0
    q_m_per_year = 10.0
    d_cond_m2_per_s = 1.0e-7
    res6_m_per_px = 2.0
    n6_count = 4
    field6 = torch.stack(
        [
            torch.full((h_px, w_px), q_m_per_year, device=dev),
            torch.zeros((h_px, w_px), device=dev),
        ]
    ).unsqueeze(0)
    pos6_px = torch.full((n6_count, 2), float(w_px // 2), device=dev)
    uL_default6 = torch.zeros((n6_count, 2), device=dev)
    uL_default6[:, 0] = 1.0
    _, dL6_px2_per_s, dT6_px2_per_s, _, _ = local_transport_state(
        pos6_px,
        field6,
        2.0 / float(w_px - 1),
        2.0 / float(h_px - 1),
        res6_m_per_px,
        SECONDS_PER_YEAR_S,
        1.0 / porosity_frac,
        1.0 / retardation_dimless,
        d_cond_m2_per_s,
        alphaL_m,
        0.0,
        res6_m_per_px**2,
        uL_default6,
    )
    v_seepage_m_per_s = (q_m_per_year / SECONDS_PER_YEAR_S) / porosity_frac
    dL_expected_px2_per_s = (
        d_cond_m2_per_s + alphaL_m * v_seepage_m_per_s / retardation_dimless
    ) / res6_m_per_px**2
    dT_expected_px2_per_s = d_cond_m2_per_s / res6_m_per_px**2
    assert abs(dL6_px2_per_s[0, 0].item() - dL_expected_px2_per_s) / dL_expected_px2_per_s < 1e-5, (
        f"Longitudinal dispersion scaling failed: got={dL6_px2_per_s[0, 0].item():.6e} exp={dL_expected_px2_per_s:.6e}"
    )
    assert abs(dT6_px2_per_s[0, 0].item() - dT_expected_px2_per_s) / dT_expected_px2_per_s < 1e-5, (
        f"Transverse dispersion scaling failed: got={dT6_px2_per_s[0, 0].item():.6e} exp={dT_expected_px2_per_s:.6e}"
    )

    # 7. Plume width in a uniform field: transverse spread after travelling L metres must
    # be sigma_T = sqrt(2 * alpha_T * L). This is the regression guard for the geometry
    # the ground truth is compared against (long, thin ribbon).
    alphaT7_m = 0.5
    res7_m_per_px = 5.0
    h7_px = 64
    w7_px = 512
    n7_count = 20000
    steps7_count = 400
    q7_m_per_year = 20.0
    porosity7_frac = 0.25
    retardation7_dimless = 5.0
    v_thermal7_m_per_s = (q7_m_per_year / SECONDS_PER_YEAR_S) / (porosity7_frac * retardation7_dimless)
    travel7_m = 1000.0
    dt7_s = travel7_m / v_thermal7_m_per_s / float(steps7_count)
    field7 = torch.stack(
        [
            torch.full((h7_px, w7_px), q7_m_per_year, device=dev),
            torch.zeros((h7_px, w7_px), device=dev),
        ]
    ).unsqueeze(0)
    acceptance7_hw = torch.ones((1, 1, h7_px, w7_px), device=dev, dtype=torch.float32)
    grid7_J = torch.zeros(h7_px * w7_px, device=dev)
    pos7_px = torch.zeros((n7_count, 2), device=dev)
    pos7_px[:, 0] = 2.0
    pos7_px[:, 1] = float(h7_px // 2)
    rwpt_seasonal_stream_kernel(
        grid7_J,
        pos7_px,
        field7,
        acceptance7_hw,
        torch.zeros(n7_count, dtype=torch.long, device=dev),
        torch.zeros((1, steps7_count), device=dev),
        torch.zeros(n7_count, dtype=torch.long, device=dev),
        dt7_s,
        steps7_count,
        w7_px,
        h7_px,
        res7_m_per_px,
        0.0,
        alphaT7_m,
        0.0,
        porosity7_frac,
        retardation7_dimless,
        SECONDS_PER_YEAR_S,
        n7_count,
    )
    inside7_bool = (pos7_px[:, 1] >= 0.0) & (pos7_px[:, 1] <= float(h7_px - 1))
    sigma7_empirical_m = (pos7_px[inside7_bool, 1].std() * res7_m_per_px).item()
    sigma7_analytical_m = math.sqrt(2.0 * alphaT7_m * travel7_m)
    width_err_frac = abs(sigma7_empirical_m - sigma7_analytical_m) / sigma7_analytical_m
    assert width_err_frac < tolerance_rel_frac, (
        f"Plume width failed: empirical={sigma7_empirical_m:.2f} m, "
        f"analytical={sigma7_analytical_m:.2f} m, error={width_err_frac:.4f}"
    )


if __name__ == "__main__":
    verify_rwpt_physical_validity(device="cuda" if torch.cuda.is_available() else "cpu")
    print("verify_rwpt_physical_validity: OK")