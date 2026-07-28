//! The resource guard: over-scale inhibition as a property of the
//! library, not of callers.
//!
//! Every representation in this crate allocates through this module and
//! checkpoints long kernels against it, so any over-scale computation is
//! inhibited *automatically*, at the point where it would actually
//! exceed the machine — not by a precomputed width constant:
//!
//! * **Memory admission** — a large allocation is admitted against the
//!   **measured** capacity at that moment: the tighter of the cgroup
//!   memory limit (v2 `memory.max` / v1 `limit_in_bytes`, minus current
//!   usage) and `/proc/meminfo MemAvailable`, with a small headroom
//!   margin. An inadmissible request fails with
//!   [`Error::OutOfMemory`] carrying the requested and available byte
//!   counts — a measurement, not a guess. Admission is backstopped by
//!   fallible allocation (`try_reserve`), so allocator refusal surfaces
//!   as the same error instead of an abort. [`set_memory_limit`]
//!   overrides the measurement with an explicit budget (research runs,
//!   tests); `None` restores auto-measurement.
//!
//!   Honesty note: on overcommitting kernels a successful reservation
//!   is still a promise, not pages; admission-by-measurement plus
//!   fallible reservation is the strongest guarantee a userspace
//!   library can give. `examples/capacity_probe.rs` verifies the
//!   guard's refusals against real OOM kills in isolated subprocesses.
//!
//! * **Time budget** — [`set_time_budget`] arms a wall-clock budget.
//!   Run entry points ([`crate::circuit::BoundCircuit::run`],
//!   [`crate::schedule::Schedule::run_on`], `Simulator::apply`,
//!   `ExactState::run`) open a deadline scope, and the long kernels
//!   (dense/sparse/exact apply sweeps, the Jacobi SVD, hierarchical
//!   merges) checkpoint against it *inside* their loops — an over-scale
//!   run aborts mid-gate with [`Error::Timeout`] carrying the budget
//!   and measured elapsed time, rather than being skipped up front on a
//!   cost estimate. A timed-out state is torn mid-operation and must be
//!   discarded, exactly like a state that observed any other error from
//!   `apply`. The default is no budget (interactive use); arming it is
//!   one call.
//!
//! The width constants that used to double as capacity policy
//! (`DENSE_MAX_QUBITS` and friends) are now **structural** bounds only
//! (basis indices fit `u64`); scale inhibition comes from here.

use std::cell::Cell;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::{Duration, Instant};

use crate::error::{Error, Result};

/// Sentinel for "no explicit limit: measure the machine".
const AUTO: u64 = u64::MAX;

/// Explicit memory limit in bytes; [`AUTO`] means measure.
static MEMORY_LIMIT: AtomicU64 = AtomicU64::new(AUTO);

/// Time budget in nanoseconds; 0 means none.
static TIME_BUDGET_NANOS: AtomicU64 = AtomicU64::new(0);

thread_local! {
    /// The active deadline scope: (armed at, deadline).
    static DEADLINE: Cell<Option<(Instant, Instant)>> = const { Cell::new(None) };
}

/// Fraction of measured availability the guard will admit (headroom for
/// the allocator, the rest of the process, and measurement skew).
const ADMIT_NUM: u128 = 15;
const ADMIT_DEN: u128 = 16;

/// Override the memory budget with an explicit byte limit, or restore
/// auto-measurement with `None`. Process-global.
pub fn set_memory_limit(limit: Option<usize>) {
    MEMORY_LIMIT.store(limit.map_or(AUTO, |v| v as u64), Ordering::SeqCst);
}

/// The explicit memory limit, if one is set (`None` = auto-measured).
pub fn memory_limit() -> Option<usize> {
    match MEMORY_LIMIT.load(Ordering::SeqCst) {
        AUTO => None,
        v => Some(v as usize),
    }
}

/// Arm (or clear) the wall-clock budget applied to each subsequently
/// entered run scope. Process-global.
pub fn set_time_budget(budget: Option<Duration>) {
    let nanos = budget.map_or(0, |d| d.as_nanos().min(u64::MAX as u128) as u64);
    TIME_BUDGET_NANOS.store(nanos, Ordering::SeqCst);
}

/// The armed wall-clock budget, if any.
pub fn time_budget() -> Option<Duration> {
    match TIME_BUDGET_NANOS.load(Ordering::SeqCst) {
        0 => None,
        n => Some(Duration::from_nanos(n)),
    }
}

/// Read a byte count from the first `/proc`-style line matching `key`
/// (kB units), if the file and key exist.
fn proc_kb(path: &str, key: &str) -> Option<usize> {
    let text = std::fs::read_to_string(path).ok()?;
    let line = text.lines().find(|l| l.starts_with(key))?;
    let kb: usize = line.split_whitespace().nth(1)?.parse().ok()?;
    Some(kb * 1024)
}

/// Read a plain byte count from a cgroup file ("max" → `None`).
fn cgroup_bytes(path: &str) -> Option<usize> {
    let text = std::fs::read_to_string(path).ok()?;
    let text = text.trim();
    if text == "max" {
        return None;
    }
    text.parse().ok()
}

/// Bytes measured as available to this process **right now**: the
/// tighter of the cgroup headroom (limit − current usage, v2 then v1)
/// and `MemAvailable`. Platforms exposing neither report `usize::MAX`
/// (admission then relies on fallible reservation alone).
pub fn measured_available() -> usize {
    let mut available = usize::MAX;
    if let Some(mem) = proc_kb("/proc/meminfo", "MemAvailable:") {
        available = available.min(mem);
    }
    // cgroup v2.
    if let Some(limit) = cgroup_bytes("/sys/fs/cgroup/memory.max") {
        let used = cgroup_bytes("/sys/fs/cgroup/memory.current").unwrap_or(0);
        available = available.min(limit.saturating_sub(used));
    }
    // cgroup v1.
    if let Some(limit) = cgroup_bytes("/sys/fs/cgroup/memory/memory.limit_in_bytes") {
        // v1 reports an enormous number when unlimited; ignore those.
        if limit < (1usize << 60) {
            let used = cgroup_bytes("/sys/fs/cgroup/memory/memory.usage_in_bytes").unwrap_or(0);
            available = available.min(limit.saturating_sub(used));
        }
    }
    available
}

/// The byte budget admission checks against right now: the explicit
/// limit if set, else the measured availability with headroom.
pub fn admission_budget() -> usize {
    match memory_limit() {
        Some(limit) => limit,
        None => {
            let avail = measured_available();
            if avail == usize::MAX {
                usize::MAX
            } else {
                ((avail as u128 * ADMIT_NUM) / ADMIT_DEN) as usize
            }
        }
    }
}

/// Admit a prospective allocation of `bytes`, or fail with the measured
/// numbers.
pub fn admit(bytes: usize, what: &str) -> Result<()> {
    let budget = admission_budget();
    if bytes > budget {
        return Err(Error::OutOfMemory {
            requested: bytes,
            available: budget,
            what: what.to_string(),
        });
    }
    Ok(())
}

/// Admission-checked, fallibly-allocated zero-filled vector: the one
/// path every large state allocation in the crate goes through.
pub(crate) fn try_vec<T: Clone>(len: usize, fill: T, what: &str) -> Result<Vec<T>> {
    let bytes = len
        .checked_mul(std::mem::size_of::<T>())
        .ok_or_else(|| Error::OutOfMemory {
            requested: usize::MAX,
            available: admission_budget(),
            what: what.to_string(),
        })?;
    admit(bytes, what)?;
    let mut v: Vec<T> = Vec::new();
    v.try_reserve_exact(len).map_err(|_| Error::OutOfMemory {
        requested: bytes,
        available: admission_budget(),
        what: what.to_string(),
    })?;
    v.resize(len, fill);
    Ok(v)
}

/// Admission check for growing an existing container by `bytes` (e.g. a
/// sparse map about to reserve); pure admission, no allocation.
pub(crate) fn admit_growth(bytes: usize, what: &str) -> Result<()> {
    admit(bytes, what)
}

/// RAII deadline scope: arms `now + budget` if a budget is configured
/// and no scope is already active (the outermost scope owns the
/// deadline). Dropping the arming scope clears it.
pub struct DeadlineScope {
    armed_here: bool,
}

/// Enter a run scope (see module docs). Cheap when no budget is set.
pub fn enter() -> DeadlineScope {
    let nanos = TIME_BUDGET_NANOS.load(Ordering::Relaxed);
    if nanos == 0 {
        return DeadlineScope { armed_here: false };
    }
    DEADLINE.with(|d| {
        if d.get().is_some() {
            DeadlineScope { armed_here: false }
        } else {
            let now = Instant::now();
            d.set(Some((now, now + Duration::from_nanos(nanos))));
            DeadlineScope { armed_here: true }
        }
    })
}

impl Drop for DeadlineScope {
    fn drop(&mut self) {
        if self.armed_here {
            DEADLINE.with(|d| d.set(None));
        }
    }
}

/// Run `f` under a wall-clock budget armed directly on THIS thread —
/// independent of (and composing with) the process-wide
/// [`set_time_budget`]: the outermost scope owns the deadline, nothing
/// global is touched, and concurrent threads are unaffected. The
/// deadline clears when the scope drops, unwinding included.
pub fn with_time_budget<T>(budget: Duration, f: impl FnOnce() -> T) -> T {
    let _scope = DeadlineScope {
        armed_here: DEADLINE.with(|d| {
            if d.get().is_some() {
                false
            } else {
                let now = Instant::now();
                d.set(Some((now, now + budget)));
                true
            }
        }),
    };
    f()
}

/// Check the active deadline, failing with the measured elapsed time
/// once it has passed. A no-op (one thread-local read) when no scope is
/// armed. Long kernels call this at coarse intervals inside their
/// loops, so over-scale runs abort mid-operation instead of being
/// pre-skipped on an estimate.
#[inline]
pub fn checkpoint() -> Result<()> {
    DEADLINE.with(|d| match d.get() {
        None => Ok(()),
        Some((armed, deadline)) => {
            let now = Instant::now();
            if now > deadline {
                Err(Error::Timeout {
                    budget_ms: (deadline - armed).as_millis() as u64,
                    elapsed_ms: (now - armed).as_millis() as u64,
                })
            } else {
                Ok(())
            }
        }
    })
}

// The guard's global-state behavior (explicit limits, armed budgets) is
// exercised in `tests/capacity.rs`, which serializes access — the lib
// unit tests here stay global-free so parallel test threads cannot
// observe a transiently tiny limit.
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measurement_reads_something_positive() {
        assert!(measured_available() > 0);
    }

    #[test]
    fn capacity_overflow_is_out_of_memory_not_a_wrap() {
        // len × size overflows usize: refused without touching any
        // global or allocator.
        let err = try_vec(usize::MAX / 4, 0u64, "overflow probe").unwrap_err();
        assert!(matches!(err, Error::OutOfMemory { .. }), "{err:?}");
    }

    #[test]
    fn checkpoint_is_free_without_a_scope() {
        assert!(checkpoint().is_ok());
    }
}
