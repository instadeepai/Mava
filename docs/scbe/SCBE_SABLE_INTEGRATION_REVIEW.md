# SCBE x Sable Integration Review

Date: 2026-03-13

Scope: clean third-party review of `instadeepai/Mava` Sable surfaces inside the fork `issdandavis/Mava`

Status: initial review and integration map

## Repo Hygiene

- Fork: `issdandavis/Mava`
- Upstream: `instadeepai/Mava`
- Local clean clone: `C:\Users\issda\Mava`
- Default branch: `develop`
- Review branch: `docs/scbe-sable-integration-review`

Decision:

- keep upstream `develop` as a clean mirror target
- keep SCBE review and bridge work on fork-local branches
- do not vendor Sable into the main SCBE repo while the SCBE tree is already noisy

That separation gives us a stable place to verify integration points without colliding with unrelated SCBE work.

## Findings

### 1. Runtime hidden-state mutation is centralized and observable

The main runtime hook is `SableNetwork.get_actions(...)` in [mava/networks/sable_network.py](C:\Users\issda\Mava\mava\networks\sable_network.py):456-479.

What happens there:

- all hidden states are decayed once per timestep
- the encoder consumes the decayed encoder state
- the decoder consumes the decayed decoder states
- the function returns a canonical `HiddenStates` tuple

That is the cleanest non-invasive SCBE observation point for inference-time trajectories.

### 2. Sable already exposes a stable hidden-state object

The canonical runtime state is `HiddenStates` in [mava/systems/sable/types.py](C:\Users\issda\Mava\mava\systems\sable\types.py):33-39:

- `encoder`
- `decoder_self_retn`
- `decoder_cross_retn`

The initialization shape is fixed by [mava/networks/utils/sable/get_init_hstates.py](C:\Users\issda\Mava\mava\networks\utils\sable\get_init_hstates.py):20-42.

This is useful because SCBE can build deterministic telemetry over one object type instead of reverse-engineering multiple ad hoc tensors.

### 3. Episode boundaries are enforced outside the network, not inside the hidden-state object

Anakin recurrent Sable resets hidden states after `done` in [mava/systems/sable/anakin/rec_sable.py](C:\Users\issda\Mava\mava\systems\sable\anakin\rec_sable.py):107-116.

Sebulba recurrent Sable does the same in [mava/systems/sable/sebulba/rec_sable.py](C:\Users\issda\Mava\mava\systems\sable\sebulba\rec_sable.py):156-173.

That means any SCBE temporal verifier must treat:

- monotone in-episode progression
- explicit reset discontinuities at episode boundaries

as different cases. If we ignore that, we will flag legal resets as temporal disorder.

### 4. Training and inference follow different memory paths

Training path:

- chunkwise encoder uses `dones` in [mava/networks/utils/sable/encode.py](C:\Users\issda\Mava\mava\networks\utils\sable\encode.py):27-55
- chunkwise decoder uses `dones` in [mava/networks/utils/sable/decode.py](C:\Users\issda\Mava\mava\networks\utils\sable\decode.py):36-83
- learner reruns the network from copied pre-rollout hidden state in [mava/systems/sable/anakin/rec_sable.py](C:\Users\issda\Mava\mava\systems\sable\anakin\rec_sable.py):120-123 and :186-194

Inference path:

- recurrent encoder consumes `decayed_hstate` in [mava/networks/utils/sable/encode.py](C:\Users\issda\Mava\mava\networks\utils\sable\encode.py):58-84
- recurrent decoder advances autoregressively in [mava/networks/utils/sable/decode.py](C:\Users\issda\Mava\mava\networks\utils\sable\decode.py):111-153

This asymmetry matters. A clean SCBE integration should start with telemetry over inference-time state transitions and only then extend to training-time trajectory replay.

### 5. The retention core is mathematically narrow enough to instrument

The most sensitive math surface is [mava/networks/retention.py](C:\Users\issda\Mava\mava\networks\retention.py):78-100 and :102-115.

The core operations are:

- hidden-state update
- decay-matrix masking around `done`
- recurrent accumulation

That is where SCBE-style path witnesses can later attach if we choose to verify retention evolution directly instead of only inspecting higher-level `HiddenStates`.

## Review Verdict

Sable is a good candidate for SCBE integration, but only if the first pass is telemetry-only.

The right interpretation is:

- Sable remains the learning and execution architecture
- SCBE observes hidden-state trajectories and scores coherence, drift, and reset legality

The wrong interpretation would be:

- replacing Sable internals with SCBE math inside the first pass
- mixing third-party code ingestion directly into the already-dirty SCBE repo

## Recommended Integration Order

### Phase 0: Mirror and freeze

Goal:

- maintain `issdandavis/Mava` as a clean fork of upstream

Rules:

- pull upstream into `develop`
- keep SCBE notes and bridge experiments on feature branches
- do not edit core Sable math on `develop`

### Phase 1: Telemetry-only bridge

Goal:

- export deterministic hidden-state summaries without affecting actions

Candidate surfaces:

- [mava/networks/sable_network.py](C:\Users\issda\Mava\mava\networks\sable_network.py):456-479
- [mava/systems/sable/anakin/rec_sable.py](C:\Users\issda\Mava\mava\systems\sable\anakin\rec_sable.py):95-116

Recommended outputs per step:

- hidden-state norms for encoder / decoder self / decoder cross
- per-step delta magnitude between previous and updated hidden states
- done/reset marker
- step count
- batch/env identifiers

This is the first safe SCBE entry point.

### Phase 2: Offline SCBE verifier

Goal:

- run SCBE checks over recorded hidden-state trajectories without touching live policy execution

Recommended checks:

- temporal monotonicity over step count
- reset legality at episode boundaries
- bounded hidden-state delta growth
- coherence scoring between encoder and decoder state families

This is where a minimal L11-style temporal verifier can be trialed without risking training behavior.

### Phase 3: Advisory runtime governance

Goal:

- emit warnings, flags, or audit packets during rollout and evaluation

Behavior:

- no action blocking yet
- no policy mutation yet
- only telemetry, thresholds, and audit traces

This preserves usability while letting us learn where the real drift signatures live.

### Phase 4: Optional gated response

Only after Phases 1-3 are stable:

- advisory quarantine tags
- evaluation-time gating
- optional runtime policy hooks

This should not be the first implementation pass.

## Recommended Initial Target

Start with recurrent Anakin Sable only:

- [mava/systems/sable/anakin/rec_sable.py](C:\Users\issda\Mava\mava\systems\sable\anakin\rec_sable.py)
- [mava/networks/sable_network.py](C:\Users\issda\Mava\mava\networks\sable_network.py)

Why:

- it is the clearest path from hidden-state update to rollout loop
- it already exposes explicit reset behavior
- it avoids mixing in Sebulba threading and queue mechanics too early

Sebulba can be mapped after the Anakin path is stable.

## Suggested Branching Model

- `develop`: upstream mirror only
- `docs/*`: review and planning notes
- `scbe/telemetry-*`: non-invasive hidden-state export
- `scbe/offline-verifier-*`: replay and scoring experiments
- `scbe/runtime-advisory-*`: live warnings only

That keeps third-party source, review work, and actual bridge code from stepping on each other.

## Immediate Next Move

Implement a tiny telemetry shim that records, per step:

- previous `HiddenStates`
- decayed `HiddenStates`
- updated `HiddenStates`
- `done`
- `step_count`

and writes the summary to a structured JSON record for offline SCBE analysis.

That is the smallest meaningful bridge between Mava/Sable and SCBE.
