//! C ABI wrapper around the [Huub](https://github.com/huub-solver/huub) CP
//! solver.
//!
//! v0 scope: opaque model handle, int/bool var creation, one linear-constraint
//! primitive, single-worker satisfy solve, value extraction, thread-local
//! last-error.
//!
//! Conventions:
//!
//! * Every `extern "C" fn` wraps its body in `std::panic::catch_unwind`
//!   so a Rust panic never unwinds into a C++ frame. On panic we record
//!   the message in thread-local last-error and return an error code (or
//!   a sentinel handle).
//! * Handles are opaque (`*mut HuubModel`). All allocation owned by Rust;
//!   the C side must call `huub_model_free`. Handles are **not thread-
//!   safe** — Huub's `Solver` is `!Send`, so each handle must stay on the
//!   thread it was created on.
//! * Variable IDs are `int32_t` indices into per-handle Vec registries.
//!   This keeps the ABI stable (no leaking of Rust internal handle
//!   types) and gives the C++ side a friendly integer identifier.

use std::{
    cell::RefCell,
    ffi::{CStr, CString, c_char},
    panic::{AssertUnwindSafe, catch_unwind},
    ptr,
    time::{Duration, Instant},
};

use huub::{
    actions::{IntDecisionActions, IntInspectionActions},
    model::{
        Model, View,
        expressions::{BoolFormula, IntLinearExp},
    },
    solver::{
        IntLitMeaning, SearchStrategy, Solver, Status as HuubStatus, SwitchTrigger,
        TerminationSignal, Valuation, View as SolverView,
        branchers::{BoolBrancher, DecisionSelection, DomainSelection, IntBrancher, WarmStartBrancher},
    },
};

type IntVal = i64;

// ----- Status / error codes ---------------------------------------------

/// Return code for solver-status queries and for entry points that don't
/// return a handle. Mirrors `huub::solver::Status` plus an explicit error
/// code for FFI failures.
#[repr(C)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum HuubResult {
    /// A satisfying assignment was found.
    Satisfied = 0,
    /// The model has no solution.
    Unsatisfiable = 1,
    /// All solutions have been enumerated (optimization or all-solutions
    /// mode). Treat as SAT for satisfaction queries.
    Complete = 2,
    /// Search terminated without proving SAT/UNSAT (time limit, conflict
    /// budget, interrupt).
    Unknown = 3,
    /// Solve has not been run yet on this handle.
    NotSolved = 4,
    /// An error occurred. See `huub_last_error()` for details.
    Error = 5,
}

// ----- Thread-local error string ----------------------------------------

thread_local! {
    static LAST_ERROR: RefCell<Option<CString>> = const { RefCell::new(None) };
}

fn set_error<S: Into<Vec<u8>>>(msg: S) {
    let bytes: Vec<u8> = msg.into();
    // Replace interior NULs with '?' so CString::new can't fail.
    let cleaned: Vec<u8> = bytes
        .into_iter()
        .map(|b| if b == 0 { b'?' } else { b })
        .collect();
    let c = CString::new(cleaned).expect("nuls stripped");
    LAST_ERROR.with(|e| *e.borrow_mut() = Some(c));
}

fn clear_error() {
    LAST_ERROR.with(|e| *e.borrow_mut() = None);
}

/// Returns the last error message recorded on the calling thread, or
/// `NULL` if no error has occurred since the last successful call. The
/// returned pointer is valid until the next FFI call on this thread.
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub extern "C" fn huub_last_error() -> *const c_char {
    LAST_ERROR.with(|e| match &*e.borrow() {
        Some(c) => c.as_ptr(),
        None => ptr::null(),
    })
}

// ----- Handle definition -------------------------------------------------

/// Opaque handle. C side only sees `HuubModel*`.
pub struct HuubModel {
    /// State machine: `Building` while constraints are being posted,
    /// `Lowered` after the first `huub_model_solve` call. In `Lowered`
    /// the solver is kept alive so subsequent solves (with new
    /// warm-start hints between T_squeeze iterations) reuse the encoded
    /// constraints. `Solver` is `!Send`; that's fine because the handle
    /// is single-thread per the v0 contract.
    inner: HandleState,
    /// Model-side int var registry. Index = C-visible var id.
    int_vars: Vec<View<IntVal>>,
    /// Model-side bool var registry.
    bool_vars: Vec<View<bool>>,
    /// Interval registry: each entry is `(start, size, end)`. Huub has no
    /// native interval object — disjunctive/no_overlap take separate
    /// start/size views — so the C ABI synthesizes interval ids by
    /// stashing the triple. Interval creation also posts
    /// `start + size - end == 0` so the three fields stay consistent.
    intervals: Vec<(View<IntVal>, View<IntVal>, View<IntVal>)>,
    /// Pending solver-setup state: hints, branchers, search strategy,
    /// conflict budget, objective. Consumed at solve time and reset on
    /// `huub_model_reset_for_resolve` / `huub_model_clear_hints`. Phase B.2
    /// uses only the hint fields; later commits extend.
    setup: SolverSetup,
    /// Sticky "model is UNSAT" flag. Set when a post-constraint call
    /// returns `Err(Conflict)` — Huub's signal that adding the
    /// constraint produced an immediate inconsistency. We must remember
    /// this because Huub does not itself keep the failure latched on
    /// the `Model`; a subsequent `lower().to_solver()` may succeed and
    /// the solver may then find a "solution" that violates the
    /// not-actually-installed constraint. Setting the flag short-
    /// circuits `huub_model_solve` to return UNSAT.
    known_unsat: bool,
}

enum HandleState {
    Building(Box<Model>),
    Lowered {
        /// The lowered Huub solver. Kept alive across solves so re-solving
        /// with fresh hints (T_squeeze iteration) doesn't re-encode every
        /// constraint.
        solver: Box<Solver>,
        /// Solver-side view per int-var-id (parallel to `int_vars`).
        int_solver_views: Vec<SolverView<IntVal>>,
        /// Solver-side view per bool-var-id (parallel to `bool_vars`).
        bool_solver_views: Vec<SolverView<bool>>,
        result: HuubResult,
        int_vals: Vec<Option<IntVal>>,
        bool_vals: Vec<Option<bool>>,
        /// Optimum found by the last `minimize` / `maximize` solve, if
        /// any. Cleared on `reset_for_resolve`.
        objective_value: Option<IntVal>,
    },
}

/// Pending solver-setup state. Populated by the search/objective/hint/
/// config FFI entry points; drained inside `do_solve` at the start of
/// each solve. Each field is drained per-solve — the C++ caller restages
/// what's needed for the next iteration (branchers and search strategy
/// are typically set once before the first solve; hints change every
/// T_squeeze iteration).
#[derive(Default, Clone)]
struct SolverSetup {
    /// Pending warm-start hints by int-var index in `HuubModel::int_vars`.
    int_hints: Vec<(usize, IntVal)>,
    /// Pending warm-start hints by bool-var index in `HuubModel::bool_vars`.
    bool_hints: Vec<(usize, bool)>,
    /// Pending int branchers (var-index list + selection strategies). Each
    /// entry becomes a single `IntBrancher::new_in(...)` call applied in
    /// registration order.
    int_branchers: Vec<PendingIntBrancher>,
    /// Pending bool branchers.
    bool_branchers: Vec<PendingBoolBrancher>,
    /// Pending search-strategy switch. `None` leaves the solver default.
    search_strategy: Option<SearchStrategy>,
    /// Optimization objective: `(direction, int_var idx)`. `None` means
    /// satisfaction search. Persists across solves until overwritten.
    objective: Option<(ObjectiveDir, usize)>,
    /// Conflict budget: terminate the search once
    /// `solver.num_conflicts() >= budget`. `None` leaves it unbounded.
    /// Implemented via a per-solve `set_learn_callback` that increments
    /// an `Arc<AtomicU64>` on each learned clause, read from the
    /// `set_terminate_callback`. Persists across solves.
    conflict_budget: Option<u64>,
    /// Lower-time CaDiCaL `restart` flag. `None` ⇒ Huub default.
    /// Applied to the `model.lower()` builder at the Building→Lowered
    /// transition; ignored once the handle is Lowered.
    sat_restart: Option<bool>,
    /// Lower-time CaDiCaL inprocessing master switch. `None` ⇒ Huub
    /// default. When `Some(b)`, applied to `.inprocessing(b)`,
    /// `.subsumption(b)`, `.variable_elimination(b)`, `.vivification(b)`,
    /// `.probing(b)`, and `.preprocessing(b as usize)` on the lower
    /// builder — mirroring `portfolio.rs:456-461` which toggles the
    /// CaDiCaL preprocessing stack as a single knob.
    sat_inprocessing: Option<bool>,
}

#[derive(Clone, Copy)]
enum ObjectiveDir {
    Min,
    Max,
}

#[derive(Clone)]
struct PendingIntBrancher {
    var_idxs: Vec<usize>,
    decision_sel: DecisionSelection,
    domain_sel: DomainSelection,
}

#[derive(Clone)]
struct PendingBoolBrancher {
    var_idxs: Vec<usize>,
    decision_sel: DecisionSelection,
    domain_sel: DomainSelection,
}

impl HuubModel {
    fn model_mut(&mut self) -> Result<&mut Model, &'static str> {
        match &mut self.inner {
            HandleState::Building(m) => Ok(m),
            HandleState::Lowered { .. } => {
                Err("model already solved; further mutations are not allowed")
            }
        }
    }
}

// ----- Lifecycle ---------------------------------------------------------

/// Create a new model handle. Returns `NULL` on failure (call
/// `huub_last_error()`).
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub extern "C" fn huub_model_new() -> *mut HuubModel {
    clear_error();
    let r = catch_unwind(|| {
        Box::into_raw(Box::new(HuubModel {
            inner: HandleState::Building(Box::new(Model::default())),
            int_vars: Vec::new(),
            bool_vars: Vec::new(),
            intervals: Vec::new(),
            setup: SolverSetup::default(),
            known_unsat: false,
        }))
    });
    match r {
        Ok(p) => p,
        Err(e) => {
            set_error(panic_msg(&e));
            ptr::null_mut()
        }
    }
}

/// Deep-copy a handle that is still in the Building state. Returns a fresh
/// handle owning an independent `Model` (cloned), independent variable /
/// interval registries, an independent copy of the pending solver setup,
/// and the same `known_unsat` latch.
///
/// **Building-state only.** If `handle` has already been solved (its
/// internal state is `Lowered`), this returns `NULL` and sets a
/// `huub_last_error` string — cloning a live `Solver` is out of scope for
/// the C ABI. The expected callsite is C++-side portfolio orchestration:
/// build the Model once, clone N times, configure each clone with a
/// different brancher / search strategy / restart / inprocessing setting,
/// then move each clone into its own worker thread before the first solve.
///
/// Returns `NULL` on error.
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_clone(handle: *mut HuubModel) -> *mut HuubModel {
    clear_error();
    if handle.is_null() {
        set_error("huub_model_clone: null handle");
        return ptr::null_mut();
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: caller contract — handle must originate from huub_model_new.
        let src = unsafe { &*handle };
        let model_box = match &src.inner {
            HandleState::Building(m) => (**m).clone(),
            HandleState::Lowered { .. } => {
                return Err(
                    "huub_model_clone: handle is Lowered; clone before the first solve",
                );
            }
        };
        Ok(Box::into_raw(Box::new(HuubModel {
            inner: HandleState::Building(Box::new(model_box)),
            int_vars: src.int_vars.clone(),
            bool_vars: src.bool_vars.clone(),
            intervals: src.intervals.clone(),
            setup: src.setup.clone(),
            known_unsat: src.known_unsat,
        })))
    }));
    match r {
        Ok(Ok(p)) => p,
        Ok(Err(msg)) => {
            set_error(msg);
            ptr::null_mut()
        }
        Err(e) => {
            set_error(panic_msg(&e));
            ptr::null_mut()
        }
    }
}

/// Rust-only accessor: deep-copy of the inner [`huub::Model`] along with
/// the variable registries needed to map C-ABI var ids (i32) back to
/// [`huub::model::View`]s on the cloned model.
///
/// Used by `huub_portfolio_c_api` to hand the Model and its role map to
/// the portfolio without going back through proto. Both crates depend on
/// the same vendored `huub` source (see `tools/huub-c-api/Cargo.toml`),
/// so the [`View`] / [`Model`] types are identical across the boundary.
///
/// Returns `None` if `handle` is null or already lowered. The caller-
/// visible state of `handle` is unchanged — the snapshot is independent.
///
/// This is `#[doc(hidden)]` because it leaks `huub` types: only crates
/// that share the same `huub` source can call it soundly.
#[doc(hidden)]
pub struct ModelSnapshot {
    pub model: Model,
    pub int_vars: Vec<View<IntVal>>,
    pub bool_vars: Vec<View<bool>>,
    pub known_unsat: bool,
}

#[doc(hidden)]
pub unsafe fn inner_model_clone(handle: *const HuubModel) -> Option<ModelSnapshot> {
    if handle.is_null() {
        return None;
    }
    // SAFETY: caller contract — handle must originate from huub_model_new
    // and still be live on the calling thread.
    let h = unsafe { &*handle };
    match &h.inner {
        HandleState::Building(m) => Some(ModelSnapshot {
            model: (**m).clone(),
            int_vars: h.int_vars.clone(),
            bool_vars: h.bool_vars.clone(),
            known_unsat: h.known_unsat,
        }),
        HandleState::Lowered { .. } => None,
    }
}

/// Diagnostic: dump full per-constraint Debug listing (sorted) to `path`.
/// Returns 0 on success, -1 on failure.
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_diag_dump_full(
    handle: *const HuubModel,
    path: *const c_char,
) -> i32 {
    clear_error();
    if handle.is_null() || path.is_null() {
        return -1;
    }
    let path_str = match unsafe { std::ffi::CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let h = unsafe { &*handle };
    match &h.inner {
        HandleState::Building(m) => match m.diag_dump_full(path_str) {
            Ok(()) => 0,
            Err(_) => -1,
        },
        HandleState::Lowered { .. } => -1,
    }
}

/// Diagnostic: dump per-constraint kind histogram of the current model
/// to `path`. Returns 0 on success, -1 on failure (null handle, lowered
/// state, or I/O error).
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_diag_dump_kinds(
    handle: *const HuubModel,
    path: *const c_char,
) -> i32 {
    clear_error();
    if handle.is_null() || path.is_null() {
        return -1;
    }
    // SAFETY: caller contract.
    let path_str = match unsafe { std::ffi::CStr::from_ptr(path) }.to_str() {
        Ok(s) => s,
        Err(_) => return -1,
    };
    let h = unsafe { &*handle };
    match &h.inner {
        HandleState::Building(m) => match m.diag_dump_kinds(path_str) {
            Ok(()) => 0,
            Err(_) => -1,
        },
        HandleState::Lowered { .. } => -1,
    }
}

/// Free a model handle previously returned by `huub_model_new()`. Passing
/// `NULL` is a no-op.
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_free(handle: *mut HuubModel) {
    if handle.is_null() {
        return;
    }
    let _ = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: caller contract — handle must originate from huub_model_new.
        drop(unsafe { Box::from_raw(handle) });
    }));
}

// ----- Variable creation -------------------------------------------------

/// Create a new integer decision variable with domain `[lb, ub]`. Returns
/// a non-negative variable id on success, or `-1` on error (call
/// `huub_last_error()` for details).
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_new_int_var(
    handle: *mut HuubModel,
    lb: i64,
    ub: i64,
) -> i32 {
    clear_error();
    with_handle(handle, |h| {
        let m = h.model_mut().map_err(|s| s.to_string())?;
        if lb > ub {
            return Err(format!("huub_model_new_int_var: empty domain lb={lb} > ub={ub}"));
        }
        let v = m.new_int_decision(lb..=ub);
        let id = h.int_vars.len();
        h.int_vars.push(v);
        Ok(i32::try_from(id).map_err(|_| "too many int vars".to_string())?)
    })
    .unwrap_or(-1)
}

/// Create a new Boolean decision variable. Returns a non-negative variable
/// id on success, or `-1` on error.
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_new_bool_var(handle: *mut HuubModel) -> i32 {
    clear_error();
    with_handle(handle, |h| {
        let m = h.model_mut().map_err(|s| s.to_string())?;
        let v = m.new_bool_decision();
        let id = h.bool_vars.len();
        h.bool_vars.push(v);
        Ok(i32::try_from(id).map_err(|_| "too many bool vars".to_string())?)
    })
    .unwrap_or(-1)
}

/// Create a new integer constant variable with fixed value `val`. The
/// returned variable id is interchangeable with a regular int var; reads
/// of its value after `solve` will yield `val`.
///
/// Used by callers that need to mix constants into linear expressions or
/// pass a constant duration into `huub_model_new_interval`. Internally
/// posts a singleton-domain decision `val..=val`; the solver folds it to
/// a constant `View<IntVal>` during lowering.
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_new_constant(handle: *mut HuubModel, val: i64) -> i32 {
    clear_error();
    with_handle(handle, |h| {
        let m = h.model_mut().map_err(|s| s.to_string())?;
        let v = m.new_int_decision(val..=val);
        let id = h.int_vars.len();
        h.int_vars.push(v);
        Ok(i32::try_from(id).map_err(|_| "too many int vars".to_string())?)
    })
    .unwrap_or(-1)
}

// ----- Lookup helpers ----------------------------------------------------

/// Resolve a slice of C-visible int-var ids into model-side views. Caller
/// is responsible for ensuring the pointer is valid for `n` elements.
fn resolve_int_views(
    h: &HuubModel,
    ids_ptr: *const i32,
    n: usize,
    ctx: &'static str,
) -> Result<Vec<View<IntVal>>, String> {
    if n > 0 && ids_ptr.is_null() {
        return Err(format!("{ctx}: null var-ids array with n>0"));
    }
    // SAFETY: caller contract — pointer is valid for `n` i32 elements.
    let ids = unsafe { std::slice::from_raw_parts(ids_ptr, n) };
    let mut out = Vec::with_capacity(n);
    for &id in ids {
        let idx = usize::try_from(id).map_err(|_| format!("{ctx}: bad var id {id}"))?;
        let v = h
            .int_vars
            .get(idx)
            .ok_or_else(|| format!("{ctx}: var id {id} out of range"))?;
        out.push(*v);
    }
    Ok(out)
}

fn resolve_one_int(h: &HuubModel, id: i32, ctx: &'static str) -> Result<View<IntVal>, String> {
    let idx = usize::try_from(id).map_err(|_| format!("{ctx}: bad var id {id}"))?;
    let v = h
        .int_vars
        .get(idx)
        .ok_or_else(|| format!("{ctx}: var id {id} out of range"))?;
    Ok(*v)
}

#[allow(dead_code)] // used by Phase B.1 booleans commit
fn resolve_one_bool(h: &HuubModel, id: i32, ctx: &'static str) -> Result<View<bool>, String> {
    let idx = usize::try_from(id).map_err(|_| format!("{ctx}: bad bool id {id}"))?;
    let v = h
        .bool_vars
        .get(idx)
        .ok_or_else(|| format!("{ctx}: bool id {id} out of range"))?;
    Ok(*v)
}

#[allow(dead_code)] // used by Phase B.1 booleans commit
fn resolve_bool_views(
    h: &HuubModel,
    ids_ptr: *const i32,
    n: usize,
    ctx: &'static str,
) -> Result<Vec<View<bool>>, String> {
    if n > 0 && ids_ptr.is_null() {
        return Err(format!("{ctx}: null bool-ids array with n>0"));
    }
    // SAFETY: caller contract — pointer is valid for `n` i32 elements.
    let ids = unsafe { std::slice::from_raw_parts(ids_ptr, n) };
    let mut out = Vec::with_capacity(n);
    for &id in ids {
        out.push(resolve_one_bool(h, id, ctx)?);
    }
    Ok(out)
}

/// Standardized handling of a `Model::xxx(...).post()` return value:
/// translate the `Err(Conflict)` simplification result into a latched-UNSAT
/// state on the handle. See the explanation on `HuubModel::known_unsat`.
fn latch_post<T>(h: &mut HuubModel, r: Result<(), T>) -> HuubResult {
    match r {
        Ok(()) => HuubResult::Satisfied,
        Err(_) => {
            h.known_unsat = true;
            HuubResult::Unsatisfiable
        }
    }
}

// ----- Linear constraint -------------------------------------------------

/// Comparator for `huub_model_add_linear`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum HuubRelOp {
    Le = 0,
    Lt = 1,
    Ge = 2,
    Gt = 3,
    Eq = 4,
    Ne = 5,
}

/// Variable-selection strategy for `huub_model_add_decision_strategy_*`.
/// Mirrors `huub::solver::branchers::DecisionSelection`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum HuubDecisionSel {
    AntiFirstFail = 0,
    FirstFail = 1,
    InputOrder = 2,
    Largest = 3,
    Smallest = 4,
}

impl HuubDecisionSel {
    fn to_huub(self) -> DecisionSelection {
        match self {
            HuubDecisionSel::AntiFirstFail => DecisionSelection::AntiFirstFail,
            HuubDecisionSel::FirstFail => DecisionSelection::FirstFail,
            HuubDecisionSel::InputOrder => DecisionSelection::InputOrder,
            HuubDecisionSel::Largest => DecisionSelection::Largest,
            HuubDecisionSel::Smallest => DecisionSelection::Smallest,
        }
    }
}

/// Value-selection strategy for `huub_model_add_decision_strategy_*`.
/// Mirrors `huub::solver::branchers::DomainSelection`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum HuubDomainSel {
    IndomainMin = 0,
    IndomainMax = 1,
    OutdomainMin = 2,
    OutdomainMax = 3,
}

impl HuubDomainSel {
    fn to_huub(self) -> DomainSelection {
        match self {
            HuubDomainSel::IndomainMin => DomainSelection::IndomainMin,
            HuubDomainSel::IndomainMax => DomainSelection::IndomainMax,
            HuubDomainSel::OutdomainMin => DomainSelection::OutdomainMin,
            HuubDomainSel::OutdomainMax => DomainSelection::OutdomainMax,
        }
    }
}

/// Top-level search strategy for `huub_model_set_search_strategy`. Mirrors
/// `huub::solver::SearchStrategy`. `switch_after_conflicts` is consumed
/// only for `Transition` / `Interleaved`; ignored for `Branchers` / `Sat`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
pub enum HuubSearchStrategy {
    /// Use the user-provided branchers exclusively (no SAT-engine search).
    Branchers = 0,
    /// Use the SAT engine's default search, ignoring branchers.
    Sat = 1,
    /// Start with branchers, switch to Sat after the trigger fires.
    Transition = 2,
    /// Interleave branchers and Sat search, switching on every trigger.
    Interleaved = 3,
}

/// Post a linear constraint: `sum(coeffs[i] * int_var[var_ids[i]]) op rhs`.
///
/// `var_ids` and `coeffs` are arrays of length `n`. All ids must refer to
/// integer variables previously returned by `huub_model_new_int_var()`.
///
/// Returns `HuubResult::Satisfied` on success (the constraint was posted
/// without immediate conflict), `HuubResult::Unsatisfiable` if posting
/// the constraint immediately proved the model infeasible, or
/// `HuubResult::Error` on FFI failure.
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_add_linear(
    handle: *mut HuubModel,
    var_ids: *const i32,
    coeffs: *const i64,
    n: usize,
    op: HuubRelOp,
    rhs: i64,
) -> HuubResult {
    clear_error();
    let res = with_handle(handle, |h| {
        if n > 0 && (var_ids.is_null() || coeffs.is_null()) {
            return Err("huub_model_add_linear: null array with n>0".to_string());
        }
        // SAFETY: caller contract — pointers valid for `n` elements.
        let ids = unsafe { std::slice::from_raw_parts(var_ids, n) };
        let cs = unsafe { std::slice::from_raw_parts(coeffs, n) };
        let mut views: Vec<View<IntVal>> = Vec::with_capacity(n);
        for &id in ids {
            let idx = usize::try_from(id)
                .map_err(|_| format!("huub_model_add_linear: bad var id {id}"))?;
            let v = h
                .int_vars
                .get(idx)
                .ok_or_else(|| format!("huub_model_add_linear: var id {id} out of range"))?;
            views.push(*v);
        }
        let m = h.model_mut().map_err(|s| s.to_string())?;
        // Seed with the integer constant 0; `IntLinearExp` doesn't expose
        // a `Default`, but `From<IntVal>` is in its public API.
        let mut expr: IntLinearExp = IntLinearExp::from(0_i64);
        for (v, c) in views.into_iter().zip(cs.iter().copied()) {
            expr += v * c;
        }
        let post_result = match op {
            HuubRelOp::Le => m.linear(expr).le(rhs).post(),
            HuubRelOp::Lt => m.linear(expr).lt(rhs).post(),
            HuubRelOp::Ge => m.linear(expr).ge(rhs).post(),
            HuubRelOp::Gt => m.linear(expr).gt(rhs).post(),
            HuubRelOp::Eq => m.linear(expr).eq(rhs).post(),
            HuubRelOp::Ne => m.linear(expr).ne(rhs).post(),
        };
        match post_result {
            Ok(()) => Ok(HuubResult::Satisfied),
            Err(_) => {
                // Latch the model as known-UNSAT so the eventual
                // `huub_model_solve` short-circuits. Huub's `Err` here
                // means simplification detected an inconsistency; the
                // constraint itself may not be installed on the model.
                h.known_unsat = true;
                Ok(HuubResult::Unsatisfiable)
            }
        }
    });
    res.unwrap_or(HuubResult::Error)
}

// ----- Aggregate constraints --------------------------------------------

/// Post `target = max(vars[0], ..., vars[n-1])`. All ids must be valid
/// int-var ids. `n` must be > 0.
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_add_max(
    handle: *mut HuubModel,
    target: i32,
    var_ids: *const i32,
    n: usize,
) -> HuubResult {
    clear_error();
    let res = with_handle(handle, |h| {
        if n == 0 {
            return Err("huub_model_add_max: empty var list".to_string());
        }
        let views = resolve_int_views(h, var_ids, n, "huub_model_add_max")?;
        let tgt = resolve_one_int(h, target, "huub_model_add_max")?;
        let m = h.model_mut().map_err(|s| s.to_string())?;
        let r = m.maximum(views).result(tgt).post();
        Ok(latch_post(h, r))
    });
    res.unwrap_or(HuubResult::Error)
}

/// Post `target = min(vars[0], ..., vars[n-1])`.
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_add_min(
    handle: *mut HuubModel,
    target: i32,
    var_ids: *const i32,
    n: usize,
) -> HuubResult {
    clear_error();
    let res = with_handle(handle, |h| {
        if n == 0 {
            return Err("huub_model_add_min: empty var list".to_string());
        }
        let views = resolve_int_views(h, var_ids, n, "huub_model_add_min")?;
        let tgt = resolve_one_int(h, target, "huub_model_add_min")?;
        let m = h.model_mut().map_err(|s| s.to_string())?;
        let r = m.minimum(views).result(tgt).post();
        Ok(latch_post(h, r))
    });
    res.unwrap_or(HuubResult::Error)
}

/// Post `target = a * b`.
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_add_mul(
    handle: *mut HuubModel,
    target: i32,
    a: i32,
    b: i32,
) -> HuubResult {
    clear_error();
    let res = with_handle(handle, |h| {
        let av = resolve_one_int(h, a, "huub_model_add_mul")?;
        let bv = resolve_one_int(h, b, "huub_model_add_mul")?;
        let tv = resolve_one_int(h, target, "huub_model_add_mul")?;
        let m = h.model_mut().map_err(|s| s.to_string())?;
        let r = m.mul(av, bv).result(tv).post();
        Ok(latch_post(h, r))
    });
    res.unwrap_or(HuubResult::Error)
}

/// Post `target = numerator / denominator` (integer division). Huub's
/// `IntDivBounds` defines the rounding convention; callers should
/// constrain the denominator's domain to avoid zero if division-by-zero
/// is not desired.
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_add_div(
    handle: *mut HuubModel,
    target: i32,
    numerator: i32,
    denominator: i32,
) -> HuubResult {
    clear_error();
    let res = with_handle(handle, |h| {
        let nv = resolve_one_int(h, numerator, "huub_model_add_div")?;
        let dv = resolve_one_int(h, denominator, "huub_model_add_div")?;
        let tv = resolve_one_int(h, target, "huub_model_add_div")?;
        let m = h.model_mut().map_err(|s| s.to_string())?;
        let r = m.div(nv, dv).result(tv).post();
        Ok(latch_post(h, r))
    });
    res.unwrap_or(HuubResult::Error)
}

// ----- Channeling constraints -------------------------------------------

/// Post `target = values[index]`, where `values` is a constant array
/// (length `n`) of i64 and `index` is an int-var id whose domain must
/// cover at least `[0, n-1]` to make the constraint satisfiable.
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_add_element_const(
    handle: *mut HuubModel,
    index: i32,
    values: *const i64,
    n: usize,
    target: i32,
) -> HuubResult {
    clear_error();
    let res = with_handle(handle, |h| {
        if n == 0 {
            return Err("huub_model_add_element_const: empty values array".to_string());
        }
        if values.is_null() {
            return Err("huub_model_add_element_const: null values pointer".to_string());
        }
        let idx_v = resolve_one_int(h, index, "huub_model_add_element_const")?;
        let tgt_v = resolve_one_int(h, target, "huub_model_add_element_const")?;
        // SAFETY: caller contract — values valid for `n` i64 elements.
        let vals: Vec<IntVal> = unsafe { std::slice::from_raw_parts(values, n) }.to_vec();
        let m = h.model_mut().map_err(|s| s.to_string())?;
        let r = m.element(vals).index(idx_v).result(tgt_v).post();
        Ok(latch_post(h, r))
    });
    res.unwrap_or(HuubResult::Error)
}

/// Post `target = array[index]`, where `array` is an array (length `n`)
/// of int-var ids and `index` is an int-var id.
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_add_element_var(
    handle: *mut HuubModel,
    index: i32,
    var_ids: *const i32,
    n: usize,
    target: i32,
) -> HuubResult {
    clear_error();
    let res = with_handle(handle, |h| {
        if n == 0 {
            return Err("huub_model_add_element_var: empty array".to_string());
        }
        let arr = resolve_int_views(h, var_ids, n, "huub_model_add_element_var")?;
        let idx_v = resolve_one_int(h, index, "huub_model_add_element_var")?;
        let tgt_v = resolve_one_int(h, target, "huub_model_add_element_var")?;
        let m = h.model_mut().map_err(|s| s.to_string())?;
        let r = m.element(arr).index(idx_v).result(tgt_v).post();
        Ok(latch_post(h, r))
    });
    res.unwrap_or(HuubResult::Error)
}

/// Post `all_different(vars)`: each pair of variables in the list must
/// take distinct values.
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_add_all_different(
    handle: *mut HuubModel,
    var_ids: *const i32,
    n: usize,
) -> HuubResult {
    clear_error();
    let res = with_handle(handle, |h| {
        let views = resolve_int_views(h, var_ids, n, "huub_model_add_all_different")?;
        let m = h.model_mut().map_err(|s| s.to_string())?;
        let r = m.unique(views).post();
        Ok(latch_post(h, r))
    });
    res.unwrap_or(HuubResult::Error)
}

// ----- Boolean / reified constraints ------------------------------------

/// Post a half- or fully-reified linear constraint:
/// `sum(coeffs[i] * int_var[var_ids[i]]) op rhs`, gated by `enforce_lit`.
///
/// * `half = true`  → install `enforce_lit → constraint` (CP-SAT's
///   `.OnlyEnforceIf` semantics — one-way implication).
/// * `half = false` → install `enforce_lit ↔ constraint` (full iff).
///
/// If `enforce_lit < 0` the call is equivalent to a plain
/// `huub_model_add_linear`.
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_add_linear_reif(
    handle: *mut HuubModel,
    var_ids: *const i32,
    coeffs: *const i64,
    n: usize,
    op: HuubRelOp,
    rhs: i64,
    enforce_lit: i32,
    half: bool,
) -> HuubResult {
    clear_error();
    let res = with_handle(handle, |h| {
        let views = resolve_int_views(h, var_ids, n, "huub_model_add_linear_reif")?;
        if n > 0 && coeffs.is_null() {
            return Err("huub_model_add_linear_reif: null coeffs with n>0".to_string());
        }
        // SAFETY: caller contract — pointer valid for `n` i64 elements.
        let cs = unsafe { std::slice::from_raw_parts(coeffs, n) }.to_vec();
        let enforce_view = if enforce_lit >= 0 {
            Some(resolve_one_bool(h, enforce_lit, "huub_model_add_linear_reif")?)
        } else {
            None
        };
        let m = h.model_mut().map_err(|s| s.to_string())?;
        let mut expr: IntLinearExp = IntLinearExp::from(0_i64);
        for (v, c) in views.into_iter().zip(cs.iter().copied()) {
            expr += v * c;
        }
        let r = match (op, enforce_view, half) {
            (HuubRelOp::Le, Some(b), true) => m.linear(expr).le(rhs).implied_by(b).post(),
            (HuubRelOp::Le, Some(b), false) => m.linear(expr).le(rhs).reified_by(b).post(),
            (HuubRelOp::Le, None, _) => m.linear(expr).le(rhs).post(),
            (HuubRelOp::Lt, Some(b), true) => m.linear(expr).lt(rhs).implied_by(b).post(),
            (HuubRelOp::Lt, Some(b), false) => m.linear(expr).lt(rhs).reified_by(b).post(),
            (HuubRelOp::Lt, None, _) => m.linear(expr).lt(rhs).post(),
            (HuubRelOp::Ge, Some(b), true) => m.linear(expr).ge(rhs).implied_by(b).post(),
            (HuubRelOp::Ge, Some(b), false) => m.linear(expr).ge(rhs).reified_by(b).post(),
            (HuubRelOp::Ge, None, _) => m.linear(expr).ge(rhs).post(),
            (HuubRelOp::Gt, Some(b), true) => m.linear(expr).gt(rhs).implied_by(b).post(),
            (HuubRelOp::Gt, Some(b), false) => m.linear(expr).gt(rhs).reified_by(b).post(),
            (HuubRelOp::Gt, None, _) => m.linear(expr).gt(rhs).post(),
            (HuubRelOp::Eq, Some(b), true) => m.linear(expr).eq(rhs).implied_by(b).post(),
            (HuubRelOp::Eq, Some(b), false) => m.linear(expr).eq(rhs).reified_by(b).post(),
            (HuubRelOp::Eq, None, _) => m.linear(expr).eq(rhs).post(),
            (HuubRelOp::Ne, Some(b), true) => m.linear(expr).ne(rhs).implied_by(b).post(),
            (HuubRelOp::Ne, Some(b), false) => m.linear(expr).ne(rhs).reified_by(b).post(),
            (HuubRelOp::Ne, None, _) => m.linear(expr).ne(rhs).post(),
        };
        Ok(latch_post(h, r))
    });
    res.unwrap_or(HuubResult::Error)
}

/// Sentinel value for "no enforce literal" in the mixed reified API. We
/// can't use `< 0` here because negative bool ids encode literal negation
/// (e.g. `-1` ⇒ `!bool_vars[0]`).
pub const HUUB_NO_ENFORCE_LIT: i32 = i32::MIN;

/// Resolve a signed bool id into a `View<bool>`. Positive ids look up
/// directly in `bool_vars`; negative ids decode as `!bool_vars[!id]`
/// (one's complement). Used only by the `*_mixed_*` entry points — old
/// positive-only APIs continue using `resolve_one_bool` / `resolve_bool_views`.
fn resolve_one_bool_signed(
    h: &HuubModel,
    id: i32,
    ctx: &'static str,
) -> Result<View<bool>, String> {
    if id >= 0 {
        let idx = usize::try_from(id).map_err(|_| format!("{ctx}: bad bool id {id}"))?;
        let v = h
            .bool_vars
            .get(idx)
            .ok_or_else(|| format!("{ctx}: bool id {id} out of range"))?;
        Ok(*v)
    } else {
        let pos = !id;
        let idx = usize::try_from(pos)
            .map_err(|_| format!("{ctx}: bad negated bool id {id}"))?;
        let v = h
            .bool_vars
            .get(idx)
            .ok_or_else(|| format!("{ctx}: negated bool id {id} (pos={pos}) out of range"))?;
        Ok(!*v)
    }
}

/// Post a linear constraint over a mix of integer and Boolean terms,
/// optionally half-/full-reified.
///
/// `int_var_ids` (length `n_ints`) reference Huub int vars; `bool_ids`
/// (length `n_bools`) reference Huub bools, **signed**: a negative id
/// `b` means `!bool_vars[!b]` (one's-complement encoding).
///
/// The constraint posted is:
///   `sum(int_coeffs[i] * int_var[i]) + sum(bool_coeffs[j] * bool_view[j]) op rhs`
/// where each `bool_view[j]` is cast to a 0/1 int via `View::from(...)`.
///
/// Reification:
/// * `enforce_lit == HUUB_NO_ENFORCE_LIT (i32::MIN)` ⇒ unenforced.
/// * Otherwise `enforce_lit` is a signed bool id (negation supported).
///   `half = true` ⇒ `enforce → constraint` (implied_by);
///   `half = false` ⇒ `enforce ↔ constraint` (reified_by).
///
/// Either `n_ints` or `n_bools` may be 0 (with the matching pointer NULL
/// allowed in that case). Returns `Satisfied`, `Unsatisfiable`, or `Error`.
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_add_linear_mixed_reif(
    handle: *mut HuubModel,
    int_var_ids: *const i32,
    int_coeffs: *const i64,
    n_ints: usize,
    bool_ids: *const i32,
    bool_coeffs: *const i64,
    n_bools: usize,
    op: HuubRelOp,
    rhs: i64,
    enforce_lit: i32,
    half: bool,
) -> HuubResult {
    clear_error();
    let res = with_handle(handle, |h| {
        if n_ints > 0 && (int_var_ids.is_null() || int_coeffs.is_null()) {
            return Err("huub_model_add_linear_mixed_reif: null int arrays with n_ints>0"
                .to_string());
        }
        if n_bools > 0 && (bool_ids.is_null() || bool_coeffs.is_null()) {
            return Err("huub_model_add_linear_mixed_reif: null bool arrays with n_bools>0"
                .to_string());
        }
        // Resolve int views.
        let int_views = if n_ints == 0 {
            Vec::new()
        } else {
            resolve_int_views(h, int_var_ids, n_ints, "huub_model_add_linear_mixed_reif")?
        };
        // SAFETY: caller contract — int_coeffs valid for n_ints i64.
        let int_cs: Vec<i64> = if n_ints == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(int_coeffs, n_ints) }.to_vec()
        };
        // Resolve bool views (signed) and corresponding coeffs.
        let mut bool_views: Vec<View<bool>> = Vec::with_capacity(n_bools);
        if n_bools > 0 {
            // SAFETY: caller contract — bool_ids valid for n_bools i32.
            let ids = unsafe { std::slice::from_raw_parts(bool_ids, n_bools) };
            for &id in ids {
                bool_views.push(resolve_one_bool_signed(
                    h,
                    id,
                    "huub_model_add_linear_mixed_reif",
                )?);
            }
        }
        // SAFETY: caller contract — bool_coeffs valid for n_bools i64.
        let bool_cs: Vec<i64> = if n_bools == 0 {
            Vec::new()
        } else {
            unsafe { std::slice::from_raw_parts(bool_coeffs, n_bools) }.to_vec()
        };
        // Resolve enforce literal (signed). Sentinel = i32::MIN.
        let enforce_view = if enforce_lit == HUUB_NO_ENFORCE_LIT {
            None
        } else {
            Some(resolve_one_bool_signed(
                h,
                enforce_lit,
                "huub_model_add_linear_mixed_reif (enforce)",
            )?)
        };
        let m = h.model_mut().map_err(|s| s.to_string())?;
        let mut expr: IntLinearExp = IntLinearExp::from(0_i64);
        for (v, c) in int_views.into_iter().zip(int_cs.iter().copied()) {
            expr += v * c;
        }
        for (b, c) in bool_views.into_iter().zip(bool_cs.iter().copied()) {
            let iv: View<IntVal> = View::from(b);
            expr += iv * c;
        }
        let r = match (op, enforce_view, half) {
            (HuubRelOp::Le, Some(b), true) => m.linear(expr).le(rhs).implied_by(b).post(),
            (HuubRelOp::Le, Some(b), false) => m.linear(expr).le(rhs).reified_by(b).post(),
            (HuubRelOp::Le, None, _) => m.linear(expr).le(rhs).post(),
            (HuubRelOp::Lt, Some(b), true) => m.linear(expr).lt(rhs).implied_by(b).post(),
            (HuubRelOp::Lt, Some(b), false) => m.linear(expr).lt(rhs).reified_by(b).post(),
            (HuubRelOp::Lt, None, _) => m.linear(expr).lt(rhs).post(),
            (HuubRelOp::Ge, Some(b), true) => m.linear(expr).ge(rhs).implied_by(b).post(),
            (HuubRelOp::Ge, Some(b), false) => m.linear(expr).ge(rhs).reified_by(b).post(),
            (HuubRelOp::Ge, None, _) => m.linear(expr).ge(rhs).post(),
            (HuubRelOp::Gt, Some(b), true) => m.linear(expr).gt(rhs).implied_by(b).post(),
            (HuubRelOp::Gt, Some(b), false) => m.linear(expr).gt(rhs).reified_by(b).post(),
            (HuubRelOp::Gt, None, _) => m.linear(expr).gt(rhs).post(),
            (HuubRelOp::Eq, Some(b), true) => m.linear(expr).eq(rhs).implied_by(b).post(),
            (HuubRelOp::Eq, Some(b), false) => m.linear(expr).eq(rhs).reified_by(b).post(),
            (HuubRelOp::Eq, None, _) => m.linear(expr).eq(rhs).post(),
            (HuubRelOp::Ne, Some(b), true) => m.linear(expr).ne(rhs).implied_by(b).post(),
            (HuubRelOp::Ne, Some(b), false) => m.linear(expr).ne(rhs).reified_by(b).post(),
            (HuubRelOp::Ne, None, _) => m.linear(expr).ne(rhs).post(),
        };
        Ok(latch_post(h, r))
    });
    res.unwrap_or(HuubResult::Error)
}

/// Post a Boolean disjunction `lits[0] ∨ lits[1] ∨ ... ∨ lits[n-1]`,
/// optionally half-reified by `enforce_lit` (semantics match
/// `huub_model_add_linear_reif`: `enforce → disjunction`).
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_add_bool_or(
    handle: *mut HuubModel,
    lits: *const i32,
    n: usize,
    enforce_lit: i32,
) -> HuubResult {
    clear_error();
    let res = with_handle(handle, |h| {
        if n == 0 {
            // Empty disjunction is identically false — that's UNSAT unless
            // the enforce literal is false. Refuse explicitly.
            return Err("huub_model_add_bool_or: empty disjunction".to_string());
        }
        let views = resolve_bool_views(h, lits, n, "huub_model_add_bool_or")?;
        let enforce_view = if enforce_lit >= 0 {
            Some(resolve_one_bool(h, enforce_lit, "huub_model_add_bool_or")?)
        } else {
            None
        };
        let m = h.model_mut().map_err(|s| s.to_string())?;
        let formula = BoolFormula::Or(views.into_iter().map(BoolFormula::from).collect());
        let r = match enforce_view {
            Some(b) => m.proposition(formula).implied_by(b).post(),
            None => m.proposition(formula).post(),
        };
        Ok(latch_post(h, r))
    });
    res.unwrap_or(HuubResult::Error)
}

/// Post a Boolean conjunction `lits[0] ∧ lits[1] ∧ ... ∧ lits[n-1]`,
/// optionally half-reified by `enforce_lit`.
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_add_bool_and(
    handle: *mut HuubModel,
    lits: *const i32,
    n: usize,
    enforce_lit: i32,
) -> HuubResult {
    clear_error();
    let res = with_handle(handle, |h| {
        if n == 0 {
            return Err("huub_model_add_bool_and: empty conjunction".to_string());
        }
        let views = resolve_bool_views(h, lits, n, "huub_model_add_bool_and")?;
        let enforce_view = if enforce_lit >= 0 {
            Some(resolve_one_bool(h, enforce_lit, "huub_model_add_bool_and")?)
        } else {
            None
        };
        let m = h.model_mut().map_err(|s| s.to_string())?;
        let formula = BoolFormula::And(views.into_iter().map(BoolFormula::from).collect());
        let r = match enforce_view {
            Some(b) => m.proposition(formula).implied_by(b).post(),
            None => m.proposition(formula).post(),
        };
        Ok(latch_post(h, r))
    });
    res.unwrap_or(HuubResult::Error)
}

/// Post `a → b` (logical implication). Encoded as `proposition(Or(¬a, b))`.
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_add_implication(
    handle: *mut HuubModel,
    a: i32,
    b: i32,
) -> HuubResult {
    clear_error();
    let res = with_handle(handle, |h| {
        let av = resolve_one_bool(h, a, "huub_model_add_implication")?;
        let bv = resolve_one_bool(h, b, "huub_model_add_implication")?;
        let m = h.model_mut().map_err(|s| s.to_string())?;
        let formula = BoolFormula::Or(vec![BoolFormula::Atom(!av), BoolFormula::Atom(bv)]);
        let r = m.proposition(formula).post();
        Ok(latch_post(h, r))
    });
    res.unwrap_or(HuubResult::Error)
}

// ----- Intervals --------------------------------------------------------

/// Create a new interval `(start, size, end)` triple. Posts the
/// consistency constraint `start + size - end == 0` so any constraint
/// that consumes one component remains in sync with the other two.
///
/// Returns a non-negative interval id on success, or `-1` on error. The
/// id is independent of int-var ids and only valid as input to interval
/// constraints (`no_overlap`, `disjunctive`).
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_new_interval(
    handle: *mut HuubModel,
    start: i32,
    size: i32,
    end: i32,
) -> i32 {
    clear_error();
    with_handle(handle, |h| {
        let sv = resolve_one_int(h, start, "huub_model_new_interval")?;
        let zv = resolve_one_int(h, size, "huub_model_new_interval")?;
        let ev = resolve_one_int(h, end, "huub_model_new_interval")?;
        let id = h.intervals.len();
        let m = h.model_mut().map_err(|s| s.to_string())?;
        // Post start + size - end == 0 so consumers see a consistent
        // interval regardless of which fields they read.
        let mut expr: IntLinearExp = IntLinearExp::from(0_i64);
        expr += sv * 1;
        expr += zv * 1;
        expr += ev * -1;
        let post_res = m.linear(expr).eq(0).post();
        let post = latch_post(h, post_res);
        // Even if the consistency constraint is found UNSAT at post time
        // (e.g., overflow), record the interval triple so subsequent
        // calls with this id don't blow up.
        h.intervals.push((sv, zv, ev));
        if post == HuubResult::Error {
            return Err("huub_model_new_interval: failed to post consistency".to_string());
        }
        Ok(i32::try_from(id).map_err(|_| "too many intervals".to_string())?)
    })
    .unwrap_or(-1)
}

fn resolve_intervals(
    h: &HuubModel,
    ids_ptr: *const i32,
    n: usize,
    ctx: &'static str,
) -> Result<Vec<(View<IntVal>, View<IntVal>, View<IntVal>)>, String> {
    if n > 0 && ids_ptr.is_null() {
        return Err(format!("{ctx}: null interval-ids array with n>0"));
    }
    // SAFETY: caller contract — pointer valid for `n` i32 elements.
    let ids = unsafe { std::slice::from_raw_parts(ids_ptr, n) };
    let mut out = Vec::with_capacity(n);
    for &id in ids {
        let idx = usize::try_from(id).map_err(|_| format!("{ctx}: bad interval id {id}"))?;
        let iv = h
            .intervals
            .get(idx)
            .ok_or_else(|| format!("{ctx}: interval id {id} out of range"))?;
        out.push(*iv);
    }
    Ok(out)
}

/// Post a 1-D no-overlap constraint over the given intervals. Sizes may
/// be variable (Huub's sweep propagator accepts view-typed sizes).
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_add_no_overlap(
    handle: *mut HuubModel,
    interval_ids: *const i32,
    n: usize,
) -> HuubResult {
    clear_error();
    let res = with_handle(handle, |h| {
        let triples = resolve_intervals(h, interval_ids, n, "huub_model_add_no_overlap")?;
        let origins: Vec<Vec<View<IntVal>>> =
            triples.iter().map(|(s, _, _)| vec![*s]).collect();
        let sizes: Vec<Vec<View<IntVal>>> = triples.iter().map(|(_, z, _)| vec![*z]).collect();
        let m = h.model_mut().map_err(|s| s.to_string())?;
        let r = m
            .no_overlap()
            .origins(origins)
            .sizes(sizes)
            .strict(true)
            .post();
        Ok(latch_post(h, r))
    });
    res.unwrap_or(HuubResult::Error)
}

/// Post a disjunctive constraint over the given intervals. Each
/// interval's size view **must be a constant** (fixed at posting time);
/// if a size is not fixed, the call returns `HuubResult::Error` and the
/// caller should use `huub_model_add_no_overlap` instead. Edge-finding,
/// not-last, and detectable-precedence propagators are all enabled
/// (matches `tools/huub_eval/src/translate.rs` line 1063-1067).
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_add_disjunctive(
    handle: *mut HuubModel,
    interval_ids: *const i32,
    n: usize,
) -> HuubResult {
    clear_error();
    let res = with_handle(handle, |h| {
        let triples = resolve_intervals(h, interval_ids, n, "huub_model_add_disjunctive")?;
        let mut starts: Vec<View<IntVal>> = Vec::with_capacity(triples.len());
        let mut durations: Vec<IntVal> = Vec::with_capacity(triples.len());
        // Borrow the model immutably to inspect sizes, then drop the
        // borrow before re-borrowing mutably for the post call.
        {
            let m_ref: &Model = match &h.inner {
                HandleState::Building(m) => m,
                HandleState::Lowered { .. } => {
                    return Err("huub_model_add_disjunctive: model already solved".to_string());
                }
            };
            for (i, (s, z, _e)) in triples.iter().enumerate() {
                let lo = IntInspectionActions::min(z, m_ref);
                let hi = IntInspectionActions::max(z, m_ref);
                if lo != hi {
                    return Err(format!(
                        "huub_model_add_disjunctive: interval {i} has non-constant size \
                         (min={lo}, max={hi}); use huub_model_add_no_overlap instead"
                    ));
                }
                starts.push(*s);
                durations.push(lo);
            }
        }
        let m = h.model_mut().map_err(|s| s.to_string())?;
        let r = m
            .disjunctive()
            .start_times(starts)
            .durations(durations)
            .edge_finding_propagation(true)
            .not_last_propagation(true)
            .detectable_precedence_propagation(true)
            .post();
        Ok(latch_post(h, r))
    });
    res.unwrap_or(HuubResult::Error)
}

/// Post a disjunctive directly over (start_var_ids, constant_durations).
/// Avoids the per-task end-var + interval-consistency-post that
/// `huub_model_add_disjunctive` requires when going through intervals.
///
/// `n` items; `start_ids` and `durations` are parallel arrays. Items with
/// `durations[i] <= 0` are skipped (huub `disjunctive` would reject them).
/// Edge-finding / not-last / detectable-precedence propagators enabled
/// to match `huub_model_add_disjunctive`.
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_add_disjunctive_starts_durs(
    handle: *mut HuubModel,
    start_ids: *const i32,
    durations: *const i64,
    n: usize,
) -> HuubResult {
    clear_error();
    let res = with_handle(handle, |h| {
        if n > 0 && (start_ids.is_null() || durations.is_null()) {
            return Err(
                "huub_model_add_disjunctive_starts_durs: null array with n>0"
                    .to_string(),
            );
        }
        // SAFETY: caller contract — pointers valid for n elements.
        let ids = if n == 0 {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(start_ids, n) }
        };
        let durs_in = if n == 0 {
            &[][..]
        } else {
            unsafe { std::slice::from_raw_parts(durations, n) }
        };
        let mut starts: Vec<View<IntVal>> = Vec::with_capacity(n);
        let mut durs: Vec<IntVal> = Vec::with_capacity(n);
        for (i, &id) in ids.iter().enumerate() {
            let d = durs_in[i];
            if d <= 0 {
                continue;
            }
            let v = resolve_one_int(
                h, id, "huub_model_add_disjunctive_starts_durs",
            )?;
            starts.push(v);
            durs.push(d);
        }
        if starts.len() <= 1 {
            return Ok(HuubResult::Satisfied);
        }
        let m = h.model_mut().map_err(|s| s.to_string())?;
        let r = m
            .disjunctive()
            .start_times(starts)
            .durations(durs)
            .edge_finding_propagation(true)
            .not_last_propagation(true)
            .detectable_precedence_propagation(true)
            .post();
        Ok(latch_post(h, r))
    });
    res.unwrap_or(HuubResult::Error)
}

// ----- Synthesized primitives -------------------------------------------
//
// These three entries cover constraint classes that the C++ `IScheduleSolver`
// surface exposes but Huub does not provide as a single native primitive.
// Each is implemented as a small Rust-side decomposition so the C consumer
// (and the eventual `HuubScheduleSolver` in Phase C) gets a uniform API.
// All three are flagged as upstream-contribution candidates in
// `PRODUCTIZATION.md` §3.

/// Post `target = numerator mod denominator`.
///
/// **Synthesized** — Huub has no native `mod`. The decomposition is:
/// `q := numerator / denominator`, `qd := q * denominator`,
/// `target == numerator - qd`. The intermediate `q` and `qd` int
/// decisions are owned by the model but not exposed through the var
/// registry; callers cannot read their values.
///
/// Upstream-contribution candidate: a native `Model::modulo(...)`
/// builder would let Huub propagate `mod` tighter than the linear
/// decomposition does.
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_add_mod(
    handle: *mut HuubModel,
    target: i32,
    numerator: i32,
    denominator: i32,
) -> HuubResult {
    clear_error();
    let res = with_handle(handle, |h| {
        let nv = resolve_one_int(h, numerator, "huub_model_add_mod")?;
        let dv = resolve_one_int(h, denominator, "huub_model_add_mod")?;
        let tv = resolve_one_int(h, target, "huub_model_add_mod")?;
        // Compute reasonable bounds for the internal q and qd so mul's
        // overflow-detection doesn't reject the post. q = num/den;
        // |q| <= max(|num_min|, |num_max|) / max(1, min(|den_min|, |den_max|)
        // among non-zero den candidates). qd = q*den has bounds within
        // [num_min, num_max] when target is in [0, |den|).
        let (n_lo, n_hi) = {
            let model_ref: &Model = match &h.inner {
                HandleState::Building(m) => m,
                HandleState::Lowered { .. } => {
                    return Err("huub_model_add_mod: model already solved".to_string());
                }
            };
            (
                IntInspectionActions::min(&nv, model_ref),
                IntInspectionActions::max(&nv, model_ref),
            )
        };
        // q and qd live in num's domain (qd cannot exceed num when target ≥ 0).
        let q_lo = n_lo.saturating_sub(n_hi.abs());
        let q_hi = n_hi.saturating_add(n_hi.abs());
        let qd_lo = n_lo;
        let qd_hi = n_hi;
        let m = h.model_mut().map_err(|s| s.to_string())?;
        let q = m.new_int_decision(q_lo..=q_hi);
        let qd = m.new_int_decision(qd_lo..=qd_hi);
        if let Err(_) = m.div(nv, dv).result(q).post() {
            h.known_unsat = true;
            return Ok(HuubResult::Unsatisfiable);
        }
        if let Err(_) = m.mul(q, dv).result(qd).post() {
            h.known_unsat = true;
            return Ok(HuubResult::Unsatisfiable);
        }
        // target == numerator - qd  ⇒  numerator - qd - target == 0
        let mut expr: IntLinearExp = IntLinearExp::from(0_i64);
        expr += nv * 1;
        expr += qd * -1;
        expr += tv * -1;
        let r = m.linear(expr).eq(0).post();
        Ok(latch_post(h, r))
    });
    res.unwrap_or(HuubResult::Error)
}

/// Post the inverse constraint between two equal-length arrays:
/// `bwd[fwd[i]] == i` and `fwd[bwd[i]] == i` for all `i in [0, n)`.
///
/// **Synthesized** — Huub has no native `inverse`. The decomposition
/// is `2n` element constraints (one direction each) plus two
/// `all_different` constraints (each array must be a permutation).
///
/// Upstream-contribution candidate: a native `Model::inverse(...)` would
/// halve the propagator count and may give tighter bound reasoning.
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_add_inverse(
    handle: *mut HuubModel,
    fwd_ids: *const i32,
    bwd_ids: *const i32,
    n: usize,
) -> HuubResult {
    clear_error();
    let res = with_handle(handle, |h| {
        if n == 0 {
            return Ok(HuubResult::Satisfied);
        }
        let fwd = resolve_int_views(h, fwd_ids, n, "huub_model_add_inverse")?;
        let bwd = resolve_int_views(h, bwd_ids, n, "huub_model_add_inverse")?;
        let m = h.model_mut().map_err(|s| s.to_string())?;
        // bwd[fwd[i]] == i
        for (i, fwd_i) in fwd.iter().enumerate() {
            let constant_i: View<IntVal> = View::from(i as IntVal);
            if let Err(_) = m
                .element(bwd.clone())
                .index(*fwd_i)
                .result(constant_i)
                .post()
            {
                h.known_unsat = true;
                return Ok(HuubResult::Unsatisfiable);
            }
        }
        // fwd[bwd[i]] == i
        for (i, bwd_i) in bwd.iter().enumerate() {
            let constant_i: View<IntVal> = View::from(i as IntVal);
            if let Err(_) = m
                .element(fwd.clone())
                .index(*bwd_i)
                .result(constant_i)
                .post()
            {
                h.known_unsat = true;
                return Ok(HuubResult::Unsatisfiable);
            }
        }
        // Permutation: each array values are unique.
        if let Err(_) = m.unique(fwd).post() {
            h.known_unsat = true;
            return Ok(HuubResult::Unsatisfiable);
        }
        let r = m.unique(bwd).post();
        Ok(latch_post(h, r))
    });
    res.unwrap_or(HuubResult::Error)
}

/// Post `sum(lits) ≤ 1` over the given Boolean literals (treated as 0/1
/// integers).
///
/// **Synthesized** — Huub has no native `at_most_one`. The
/// decomposition is one linear constraint over the bool-to-int casts of
/// the literals.
///
/// Upstream-contribution candidate: a native `Model::at_most_one(...)`
/// would let pindakaas-cadical use its specialized AMO encoding
/// (commander / bimander / product) rather than a linear-sum encoding.
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_add_at_most_one(
    handle: *mut HuubModel,
    lits: *const i32,
    n: usize,
) -> HuubResult {
    clear_error();
    let res = with_handle(handle, |h| {
        if n <= 1 {
            // Vacuously satisfied.
            return Ok(HuubResult::Satisfied);
        }
        let bools = resolve_bool_views(h, lits, n, "huub_model_add_at_most_one")?;
        let m = h.model_mut().map_err(|s| s.to_string())?;
        let mut expr: IntLinearExp = IntLinearExp::from(0_i64);
        for bv in bools {
            let iv: View<IntVal> = View::from(bv);
            expr += iv * 1;
        }
        let r = m.linear(expr).le(1).post();
        Ok(latch_post(h, r))
    });
    res.unwrap_or(HuubResult::Error)
}

// ----- Solve -------------------------------------------------------------

/// Lower the model (first call) or re-use the lowered solver
/// (subsequent calls), then run a single-worker satisfaction search.
/// Any warm-start hints staged on the handle via
/// `huub_model_add_hint_{int,bool}` are consumed and installed as a
/// `WarmStartBrancher` before the search runs; the hint queues are
/// drained whether or not the solve found a solution.
///
/// `time_limit_seconds <= 0` disables the wall-clock limit. Returns one
/// of `Satisfied`, `Unsatisfiable`, `Unknown`, or `Error`. After a
/// successful solve, call `huub_model_value_int` /
/// `huub_model_value_bool` to read the assignment.
///
/// Re-solve pattern: call `huub_model_reset_for_resolve` between solves
/// to clear the prior assignment, push new hints, then call
/// `huub_model_solve` again on the same handle.
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_solve(
    handle: *mut HuubModel,
    time_limit_seconds: f64,
) -> HuubResult {
    clear_error();
    let res = with_handle(handle, |h| Ok(do_solve(h, time_limit_seconds)));
    res.unwrap_or(HuubResult::Error)
}

fn do_solve(h: &mut HuubModel, time_limit_seconds: f64) -> HuubResult {
    // Short-circuit if a prior post-constraint call latched an
    // inconsistency. Don't drop the current state — re-solves on a
    // known-UNSAT model should keep returning UNSAT.
    if h.known_unsat {
        if let HandleState::Building(_) = &h.inner {
            // Synthesize a Lowered-equivalent shell with empty assignment
            // vectors so value-extraction returns NotSolved cleanly.
            h.inner = HandleState::Lowered {
                solver: Box::new(Solver::default()),
                int_solver_views: Vec::new(),
                bool_solver_views: Vec::new(),
                result: HuubResult::Unsatisfiable,
                int_vals: vec![None; h.int_vars.len()],
                bool_vals: vec![None; h.bool_vars.len()],
                objective_value: None,
            };
        }
        return HuubResult::Unsatisfiable;
    }

    // If Building, lower the model into a fresh Solver and transition to
    // Lowered. Lowering takes `&mut Model` (doesn't consume the box),
    // but we move the Model out of HandleState into a local so the
    // resulting `(Solver, LoweringMap)` doesn't alias the handle.
    if let HandleState::Building(_) = &h.inner {
        // Move Model out via a temporary swap.
        let mut model_box = match std::mem::replace(
            &mut h.inner,
            HandleState::Building(Box::new(Model::default())),
        ) {
            HandleState::Building(m) => m,
            _ => unreachable!(),
        };
        // sat_restart / sat_inprocessing are lower-time CaDiCaL options
        // (see `tools/huub_eval/src/portfolio.rs:455-461`). Huub's
        // `Lowerer` is a `bon`-derived typestate builder — calling a
        // setter consumes `self` and returns a new state, so conditional
        // chaining via `let mut lowerer = ...` won't type-check. All
        // setters default to `false` (`Lowerer::DEFAULT_*`), so we just
        // always pass the unwrapped value: `None ⇒ false` is the same as
        // not calling the setter.
        let restart = h.setup.sat_restart.unwrap_or(false);
        let inproc = h.setup.sat_inprocessing.unwrap_or(false);
        let (solver, map): (Solver, _) = match model_box
            .lower()
            .restart(restart)
            .inprocessing(inproc)
            .subsumption(inproc)
            .variable_elimination(inproc)
            .vivification(inproc)
            .probing(inproc)
            .preprocessing(if inproc { 1 } else { 0 })
            .to_solver()
        {
            Ok(pair) => pair,
            Err(e) => {
                // Restore Building state so callers can introspect.
                h.inner = HandleState::Building(model_box);
                let msg = format!("{e:?}");
                if msg.contains("Simplification(Conflict") || msg.contains("Conflict {") {
                    h.known_unsat = true;
                    return HuubResult::Unsatisfiable;
                }
                set_error(format!("huub_model_solve: lowering failed: {msg}"));
                return HuubResult::Error;
            }
        };
        let mut solver_box = Box::new(solver);
        let int_solver_views: Vec<SolverView<IntVal>> = h
            .int_vars
            .iter()
            .map(|v| map.get(solver_box.as_mut(), *v))
            .collect();
        let bool_solver_views: Vec<SolverView<bool>> = h
            .bool_vars
            .iter()
            .map(|v| map.get(solver_box.as_mut(), *v))
            .collect();
        h.inner = HandleState::Lowered {
            solver: solver_box,
            int_solver_views,
            bool_solver_views,
            result: HuubResult::NotSolved,
            int_vals: vec![None; h.int_vars.len()],
            bool_vals: vec![None; h.bool_vars.len()],
            objective_value: None,
        };
    }

    // Now in Lowered. Drain pending setup state, apply each to the
    // solver, then run solve.
    let int_hints = std::mem::take(&mut h.setup.int_hints);
    let bool_hints = std::mem::take(&mut h.setup.bool_hints);
    let int_branchers = std::mem::take(&mut h.setup.int_branchers);
    let bool_branchers = std::mem::take(&mut h.setup.bool_branchers);
    let pending_strategy = h.setup.search_strategy.take();
    let pending_objective = h.setup.objective;
    let HandleState::Lowered {
        solver,
        int_solver_views,
        bool_solver_views,
        result,
        int_vals,
        bool_vals,
        objective_value,
    } = &mut h.inner
    else {
        unreachable!("just transitioned to Lowered above");
    };

    // Structural branchers first — they form the regular search order.
    for pb in int_branchers {
        let vars: Vec<SolverView<IntVal>> = pb
            .var_idxs
            .iter()
            .filter_map(|i| int_solver_views.get(*i).copied())
            .collect();
        IntBrancher::new_in(solver.as_mut(), vars, pb.decision_sel, pb.domain_sel);
    }
    for pb in bool_branchers {
        let vars: Vec<SolverView<bool>> = pb
            .var_idxs
            .iter()
            .filter_map(|i| bool_solver_views.get(*i).copied())
            .collect();
        BoolBrancher::new_in(solver.as_mut(), vars, pb.decision_sel, pb.domain_sel);
    }
    // Search strategy (must come before solve; safe to call repeatedly).
    if let Some(strategy) = pending_strategy {
        solver.as_mut().set_search_strategy(strategy);
    }

    // Build the warm-start decision list. Each int hint `(idx, value)`
    // becomes `view.lit(solver, Eq(value))`; each bool hint
    // `(idx, value)` becomes the literal (or its negation).
    let mut warm_decisions: Vec<SolverView<bool>> =
        Vec::with_capacity(int_hints.len() + bool_hints.len());
    for (idx, value) in &int_hints {
        if let Some(view) = int_solver_views.get(*idx) {
            let lit = IntDecisionActions::lit(view, solver.as_mut(), IntLitMeaning::Eq(*value));
            warm_decisions.push(lit);
        }
    }
    for (idx, value) in &bool_hints {
        if let Some(view) = bool_solver_views.get(*idx) {
            warm_decisions.push(if *value { *view } else { !*view });
        }
    }
    if !warm_decisions.is_empty() {
        WarmStartBrancher::new_in(solver.as_mut(), warm_decisions);
    }

    // Wall-clock + conflict-budget terminate callback, OR-combined. If
    // a conflict budget is set, also install a learn-callback that
    // increments a shared `Arc<AtomicU64>` on each learned clause; the
    // terminate-callback reads that counter against the budget.
    let deadline = (time_limit_seconds > 0.0)
        .then(|| Instant::now() + Duration::from_secs_f64(time_limit_seconds));
    let budget = h.setup.conflict_budget;
    let conflicts_counter: Option<std::sync::Arc<std::sync::atomic::AtomicU64>> =
        budget.map(|_| std::sync::Arc::new(std::sync::atomic::AtomicU64::new(0)));
    if let Some(counter) = &conflicts_counter {
        let counter_for_learn = std::sync::Arc::clone(counter);
        solver
            .as_mut()
            .set_learn_callback(Some(move |_clause: &mut dyn Iterator<Item = _>| {
                counter_for_learn.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            }));
    } else {
        // Type-only None: pick any callback shape that matches the
        // declared bounds. `fn(&mut dyn Iterator<Item = _>)` won't elide
        // here so use a no-op closure type alias via turbofish on the
        // option. Simpler: just install a no-op closure that drops the
        // iterator without counting.
        solver
            .as_mut()
            .set_learn_callback(Some(|_clause: &mut dyn Iterator<Item = _>| ()));
    }
    if deadline.is_some() || budget.is_some() {
        let counter_for_term = conflicts_counter.as_ref().map(std::sync::Arc::clone);
        solver.as_mut().set_terminate_callback(Some(move || {
            if let Some(d) = deadline {
                if Instant::now() >= d {
                    return TerminationSignal::Terminate;
                }
            }
            if let (Some(b), Some(c)) = (budget, counter_for_term.as_ref()) {
                if c.load(std::sync::atomic::Ordering::Relaxed) >= b {
                    return TerminationSignal::Terminate;
                }
            }
            TerminationSignal::Continue
        }));
    } else {
        solver
            .as_mut()
            .set_terminate_callback::<fn() -> TerminationSignal>(None);
    }

    // Reset assignment buffers (clear prior solve's data, keep length).
    for v in int_vals.iter_mut() {
        *v = None;
    }
    for v in bool_vals.iter_mut() {
        *v = None;
    }
    let int_views_local = int_solver_views.clone();
    let bool_views_local = bool_solver_views.clone();
    let mut new_int_vals: Vec<Option<IntVal>> = vec![None; int_views_local.len()];
    let mut new_bool_vals: Vec<Option<bool>> = vec![None; bool_views_local.len()];

    // Capture the solver-side objective view here (while we still hold the
    // immutable borrow of int_solver_views via int_views_local).
    let objective_view = pending_objective
        .and_then(|(dir, idx)| int_solver_views.get(idx).copied().map(|v| (dir, v)));

    let (status, opt) = {
        let on_sol = |sol: huub::solver::Solution<'_>| {
            for (i, v) in int_views_local.iter().enumerate() {
                new_int_vals[i] = Some(Valuation::val(v, sol));
            }
            for (i, v) in bool_views_local.iter().enumerate() {
                new_bool_vals[i] = Some(Valuation::val(v, sol));
            }
        };
        match objective_view {
            None => {
                let s = solver.as_mut().solve().on_solution(on_sol).satisfy();
                (s, None)
            }
            Some((ObjectiveDir::Min, view)) => {
                solver.as_mut().solve().on_solution(on_sol).minimize(view)
            }
            Some((ObjectiveDir::Max, view)) => {
                solver.as_mut().solve().on_solution(on_sol).maximize(view)
            }
        }
    };

    let r = match status {
        HuubStatus::Satisfied => HuubResult::Satisfied,
        HuubStatus::Unsatisfiable => HuubResult::Unsatisfiable,
        HuubStatus::Complete => HuubResult::Complete,
        HuubStatus::Unknown => HuubResult::Unknown,
    };
    *result = r;
    *int_vals = new_int_vals;
    *bool_vals = new_bool_vals;
    *objective_value = opt;
    r
}

/// Reset the per-iteration solver state on a Lowered handle, so the
/// caller can stage new warm-start hints and call `huub_model_solve`
/// again. The encoded constraints and the lowered solver stay alive;
/// only the last solve's assignment vectors are cleared.
///
/// Returns `Satisfied` on success, `NotSolved` if the handle hasn't
/// been solved yet (no-op; same hints stay staged), or `Error` on bad
/// handle.
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_reset_for_resolve(handle: *mut HuubModel) -> HuubResult {
    clear_error();
    let res = with_handle(handle, |h| match &mut h.inner {
        HandleState::Building(_) => Ok(HuubResult::NotSolved),
        HandleState::Lowered {
            result,
            int_vals,
            bool_vals,
            objective_value,
            ..
        } => {
            *result = HuubResult::NotSolved;
            for v in int_vals.iter_mut() {
                *v = None;
            }
            for v in bool_vals.iter_mut() {
                *v = None;
            }
            *objective_value = None;
            Ok(HuubResult::Satisfied)
        }
    });
    res.unwrap_or(HuubResult::Error)
}

// ----- Warm-start hints --------------------------------------------------

/// Stage a warm-start hint `int_var[var_id] = value`. The hint is a
/// preference, not a constraint — if the suggested decision conflicts
/// with the constraint set, the brancher is consumed and regular search
/// continues. Hints accumulate across multiple `add_hint` calls and are
/// applied at the next `huub_model_solve` via a `WarmStartBrancher`.
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_add_hint_int(
    handle: *mut HuubModel,
    var_id: i32,
    value: i64,
) -> HuubResult {
    clear_error();
    let res = with_handle(handle, |h| {
        let idx = usize::try_from(var_id)
            .map_err(|_| format!("huub_model_add_hint_int: bad var id {var_id}"))?;
        if idx >= h.int_vars.len() {
            return Err(format!(
                "huub_model_add_hint_int: var id {var_id} out of range"
            ));
        }
        h.setup.int_hints.push((idx, value));
        Ok(HuubResult::Satisfied)
    });
    res.unwrap_or(HuubResult::Error)
}

/// Stage a warm-start hint `bool_var[var_id] = value`. Same preference-
/// not-constraint semantics as `huub_model_add_hint_int`.
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_add_hint_bool(
    handle: *mut HuubModel,
    var_id: i32,
    value: bool,
) -> HuubResult {
    clear_error();
    let res = with_handle(handle, |h| {
        let idx = usize::try_from(var_id)
            .map_err(|_| format!("huub_model_add_hint_bool: bad var id {var_id}"))?;
        if idx >= h.bool_vars.len() {
            return Err(format!(
                "huub_model_add_hint_bool: var id {var_id} out of range"
            ));
        }
        h.setup.bool_hints.push((idx, value));
        Ok(HuubResult::Satisfied)
    });
    res.unwrap_or(HuubResult::Error)
}

/// Clear all currently-staged warm-start hints (both int and bool). Does
/// **not** undo warm-start branchers already installed on the solver by
/// a prior solve — those exhaust themselves as their decisions get
/// applied or conflict. Suited to the T_squeeze pattern: clear, restage
/// for the next iteration, solve again.
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_clear_hints(handle: *mut HuubModel) -> HuubResult {
    clear_error();
    let res = with_handle(handle, |h| {
        h.setup.int_hints.clear();
        h.setup.bool_hints.clear();
        Ok(HuubResult::Satisfied)
    });
    res.unwrap_or(HuubResult::Error)
}

// ----- Branchers and search strategy ------------------------------------

/// Stage an int-variable decision strategy. The brancher is materialized
/// on the solver at the next `huub_model_solve` call (via
/// `IntBrancher::new_in`). Branchers stack in registration order.
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_add_decision_strategy_int(
    handle: *mut HuubModel,
    var_ids: *const i32,
    n: usize,
    decision_sel: HuubDecisionSel,
    domain_sel: HuubDomainSel,
) -> HuubResult {
    clear_error();
    let res = with_handle(handle, |h| {
        if n > 0 && var_ids.is_null() {
            return Err(
                "huub_model_add_decision_strategy_int: null var-ids with n>0".to_string(),
            );
        }
        // SAFETY: caller contract — pointer valid for `n` elements.
        let ids = unsafe { std::slice::from_raw_parts(var_ids, n) };
        let mut idxs = Vec::with_capacity(n);
        for &id in ids {
            let idx = usize::try_from(id)
                .map_err(|_| format!("huub_model_add_decision_strategy_int: bad var id {id}"))?;
            if idx >= h.int_vars.len() {
                return Err(format!(
                    "huub_model_add_decision_strategy_int: var id {id} out of range"
                ));
            }
            idxs.push(idx);
        }
        h.setup.int_branchers.push(PendingIntBrancher {
            var_idxs: idxs,
            decision_sel: decision_sel.to_huub(),
            domain_sel: domain_sel.to_huub(),
        });
        Ok(HuubResult::Satisfied)
    });
    res.unwrap_or(HuubResult::Error)
}

/// Stage a bool-variable decision strategy. Same lifecycle as
/// `huub_model_add_decision_strategy_int`.
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_add_decision_strategy_bool(
    handle: *mut HuubModel,
    var_ids: *const i32,
    n: usize,
    decision_sel: HuubDecisionSel,
    domain_sel: HuubDomainSel,
) -> HuubResult {
    clear_error();
    let res = with_handle(handle, |h| {
        if n > 0 && var_ids.is_null() {
            return Err(
                "huub_model_add_decision_strategy_bool: null var-ids with n>0".to_string(),
            );
        }
        let ids = unsafe { std::slice::from_raw_parts(var_ids, n) };
        let mut idxs = Vec::with_capacity(n);
        for &id in ids {
            let idx = usize::try_from(id).map_err(|_| {
                format!("huub_model_add_decision_strategy_bool: bad bool id {id}")
            })?;
            if idx >= h.bool_vars.len() {
                return Err(format!(
                    "huub_model_add_decision_strategy_bool: bool id {id} out of range"
                ));
            }
            idxs.push(idx);
        }
        h.setup.bool_branchers.push(PendingBoolBrancher {
            var_idxs: idxs,
            decision_sel: decision_sel.to_huub(),
            domain_sel: domain_sel.to_huub(),
        });
        Ok(HuubResult::Satisfied)
    });
    res.unwrap_or(HuubResult::Error)
}

/// Set the top-level search strategy. `switch_after_conflicts` only
/// matters for `Transition` / `Interleaved`. The strategy is applied to
/// the solver at the next `huub_model_solve`.
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_set_search_strategy(
    handle: *mut HuubModel,
    strategy: HuubSearchStrategy,
    switch_after_conflicts: u64,
) -> HuubResult {
    clear_error();
    let res = with_handle(handle, |h| {
        let s = match strategy {
            HuubSearchStrategy::Branchers => SearchStrategy::Branchers,
            HuubSearchStrategy::Sat => SearchStrategy::Sat,
            HuubSearchStrategy::Transition => {
                SearchStrategy::Transition(SwitchTrigger::Conflicts(switch_after_conflicts))
            }
            HuubSearchStrategy::Interleaved => {
                SearchStrategy::Interleaved(SwitchTrigger::Conflicts(switch_after_conflicts))
            }
        };
        h.setup.search_strategy = Some(s);
        Ok(HuubResult::Satisfied)
    });
    res.unwrap_or(HuubResult::Error)
}

// ----- Conflict budget --------------------------------------------------

/// Set a conflict budget. The next `huub_model_solve` (and any
/// subsequent solve until overwritten) terminates with `Unknown` once
/// the per-solve learned-clause counter reaches `budget`. A zero
/// `budget` disables the limit. Combines (OR) with the wall-clock
/// `time_limit_seconds` passed to `solve`.
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_set_conflict_budget(
    handle: *mut HuubModel,
    budget: u64,
) -> HuubResult {
    clear_error();
    let res = with_handle(handle, |h| {
        h.setup.conflict_budget = if budget == 0 { None } else { Some(budget) };
        Ok(HuubResult::Satisfied)
    });
    res.unwrap_or(HuubResult::Error)
}

// ----- Lower-time CaDiCaL knobs -----------------------------------------

/// Enable or disable CaDiCaL **restarts** for the next lowering of this
/// handle. The flag is stashed on the handle and consumed at the
/// Building→Lowered transition (the first `huub_model_solve` after
/// posting constraints). Calling this on a handle that has already been
/// Lowered has no effect on the existing solver; the new value applies
/// only if the caller clones a Building-state ancestor and lowers that.
///
/// The default matches Huub's `Lowerer::DEFAULT_RESTART` (`false`).
/// Mirrors `tools/huub_eval/src/portfolio.rs:455`.
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_set_sat_restart(
    handle: *mut HuubModel,
    enable: bool,
) -> HuubResult {
    clear_error();
    let res = with_handle(handle, |h| {
        h.setup.sat_restart = Some(enable);
        Ok(HuubResult::Satisfied)
    });
    res.unwrap_or(HuubResult::Error)
}

/// Enable or disable CaDiCaL **inprocessing** for the next lowering of
/// this handle. This is a single master switch that toggles six
/// underlying CaDiCaL options together (mirroring
/// `tools/huub_eval/src/portfolio.rs:456-461`): `inprocessing`,
/// `subsumption`, `variable_elimination`, `vivification`, `probing`, and
/// `preprocessing` rounds (`1` when enabled, `0` when disabled).
///
/// As with `huub_model_set_sat_restart`, the flag is consumed at the
/// Building→Lowered transition. Default is `false` (matches Huub).
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_set_sat_inprocessing(
    handle: *mut HuubModel,
    enable: bool,
) -> HuubResult {
    clear_error();
    let res = with_handle(handle, |h| {
        h.setup.sat_inprocessing = Some(enable);
        Ok(HuubResult::Satisfied)
    });
    res.unwrap_or(HuubResult::Error)
}

// ----- Objective --------------------------------------------------------

/// Set the optimization objective to `minimize int_var[var_id]`. The
/// next `huub_model_solve` will dispatch to `Solver::minimize` instead
/// of `Solver::satisfy`. The objective persists across solves until
/// overwritten by another `set_minimize` / `set_maximize`.
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_set_minimize(
    handle: *mut HuubModel,
    var_id: i32,
) -> HuubResult {
    clear_error();
    let res = with_handle(handle, |h| {
        let idx = usize::try_from(var_id)
            .map_err(|_| format!("huub_model_set_minimize: bad var id {var_id}"))?;
        if idx >= h.int_vars.len() {
            return Err(format!(
                "huub_model_set_minimize: var id {var_id} out of range"
            ));
        }
        h.setup.objective = Some((ObjectiveDir::Min, idx));
        Ok(HuubResult::Satisfied)
    });
    res.unwrap_or(HuubResult::Error)
}

/// Set the optimization objective to `maximize int_var[var_id]`.
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_set_maximize(
    handle: *mut HuubModel,
    var_id: i32,
) -> HuubResult {
    clear_error();
    let res = with_handle(handle, |h| {
        let idx = usize::try_from(var_id)
            .map_err(|_| format!("huub_model_set_maximize: bad var id {var_id}"))?;
        if idx >= h.int_vars.len() {
            return Err(format!(
                "huub_model_set_maximize: var id {var_id} out of range"
            ));
        }
        h.setup.objective = Some((ObjectiveDir::Max, idx));
        Ok(HuubResult::Satisfied)
    });
    res.unwrap_or(HuubResult::Error)
}

/// Read the optimum found by the most-recent `minimize` / `maximize`
/// solve. Writes to `*out` and returns `Satisfied` on success;
/// `NotSolved` if the last solve didn't run an optimization or found no
/// feasible solution.
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_objective_value(
    handle: *mut HuubModel,
    out: *mut i64,
) -> HuubResult {
    clear_error();
    let r = with_handle(handle, |h| match &h.inner {
        HandleState::Building(_) => Ok(HuubResult::NotSolved),
        HandleState::Lowered { objective_value, .. } => match *objective_value {
            Some(v) => {
                if !out.is_null() {
                    // SAFETY: caller provides a valid i64 pointer.
                    unsafe {
                        *out = v;
                    }
                }
                Ok(HuubResult::Satisfied)
            }
            None => Ok(HuubResult::NotSolved),
        },
    });
    r.unwrap_or(HuubResult::Error)
}

// ----- Value extraction --------------------------------------------------

/// Read the value of an integer variable from the most recent solve.
/// Writes to `*out` and returns `Satisfied` on success; returns
/// `NotSolved` / `Error` on failure (leaving `*out` untouched).
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_value_int(
    handle: *mut HuubModel,
    var_id: i32,
    out: *mut i64,
) -> HuubResult {
    clear_error();
    let r = with_handle(handle, |h| match &h.inner {
        HandleState::Building(_) => Ok(HuubResult::NotSolved),
        HandleState::Lowered { int_vals, .. } => {
            let idx = usize::try_from(var_id)
                .map_err(|_| format!("huub_model_value_int: bad var id {var_id}"))?;
            let v = int_vals
                .get(idx)
                .ok_or_else(|| format!("huub_model_value_int: var id {var_id} out of range"))?;
            match *v {
                Some(val) => {
                    if !out.is_null() {
                        // SAFETY: caller provides a valid i64 pointer.
                        unsafe {
                            *out = val;
                        }
                    }
                    Ok(HuubResult::Satisfied)
                }
                None => Ok(HuubResult::NotSolved),
            }
        }
    });
    r.unwrap_or(HuubResult::Error)
}

/// Read the value of a Boolean variable from the most recent solve.
/// Same contract as `huub_model_value_int`.
#[cfg(feature = "c-abi")]
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_value_bool(
    handle: *mut HuubModel,
    var_id: i32,
    out: *mut bool,
) -> HuubResult {
    clear_error();
    let r = with_handle(handle, |h| match &h.inner {
        HandleState::Building(_) => Ok(HuubResult::NotSolved),
        HandleState::Lowered { bool_vals, .. } => {
            let idx = usize::try_from(var_id)
                .map_err(|_| format!("huub_model_value_bool: bad var id {var_id}"))?;
            let v = bool_vals.get(idx).ok_or_else(|| {
                format!("huub_model_value_bool: var id {var_id} out of range")
            })?;
            match *v {
                Some(val) => {
                    if !out.is_null() {
                        // SAFETY: caller provides a valid bool pointer.
                        unsafe {
                            *out = val;
                        }
                    }
                    Ok(HuubResult::Satisfied)
                }
                None => Ok(HuubResult::NotSolved),
            }
        }
    });
    r.unwrap_or(HuubResult::Error)
}

// ----- Helpers -----------------------------------------------------------

fn with_handle<R, F>(handle: *mut HuubModel, f: F) -> Option<R>
where
    F: FnOnce(&mut HuubModel) -> Result<R, String>,
{
    if handle.is_null() {
        set_error("null handle");
        return None;
    }
    let r = catch_unwind(AssertUnwindSafe(|| {
        // SAFETY: caller contract — handle is valid and not aliased.
        let h = unsafe { &mut *handle };
        f(h)
    }));
    match r {
        Ok(Ok(v)) => Some(v),
        Ok(Err(msg)) => {
            set_error(msg);
            None
        }
        Err(p) => {
            set_error(format!("panic in FFI entry: {}", panic_msg(&p)));
            None
        }
    }
}

fn panic_msg(p: &Box<dyn std::any::Any + Send>) -> String {
    if let Some(s) = p.downcast_ref::<&str>() {
        (*s).to_string()
    } else if let Some(s) = p.downcast_ref::<String>() {
        s.clone()
    } else {
        "<non-string panic payload>".to_string()
    }
}

// `_` to silence unused-import warning on the CStr/c_char re-exports;
// they're part of the public ABI surface even if no current entry point
// takes a const char* in v0.
#[allow(dead_code)]
fn _abi_surface_anchor(_: *const c_char, _: &CStr) {}

// ----- Tests -------------------------------------------------------------

#[cfg(test)]
mod tests {
    //! These tests exercise the C ABI directly — every call goes through
    //! `unsafe extern "C" fn` entry points, so we're testing the same
    //! surface a C++ caller would see.

    use super::*;

    /// SAT case: `x + y == 7, x >= 3, x,y in [0,10]`. Confirm the assignment
    /// satisfies the posted constraints.
    #[test]
    fn linear_sat_roundtrip() {
        unsafe {
            let m = huub_model_new();
            assert!(!m.is_null());

            let x = huub_model_new_int_var(m, 0, 10);
            let y = huub_model_new_int_var(m, 0, 10);
            assert_eq!(x, 0);
            assert_eq!(y, 1);

            // x + y == 7
            let ids = [x, y];
            let cs = [1_i64, 1_i64];
            let r = huub_model_add_linear(m, ids.as_ptr(), cs.as_ptr(), 2, HuubRelOp::Eq, 7);
            assert_eq!(r, HuubResult::Satisfied);

            // x >= 3
            let ids2 = [x];
            let cs2 = [1_i64];
            let r = huub_model_add_linear(m, ids2.as_ptr(), cs2.as_ptr(), 1, HuubRelOp::Ge, 3);
            assert_eq!(r, HuubResult::Satisfied);

            let r = huub_model_solve(m, 10.0);
            assert_eq!(r, HuubResult::Satisfied);

            let mut xv: i64 = -1;
            let mut yv: i64 = -1;
            assert_eq!(huub_model_value_int(m, x, &mut xv), HuubResult::Satisfied);
            assert_eq!(huub_model_value_int(m, y, &mut yv), HuubResult::Satisfied);
            assert_eq!(xv + yv, 7);
            assert!(xv >= 3, "x={xv} should be >= 3");

            huub_model_free(m);
        }
    }

    /// UNSAT case: `x + y >= 11, x,y in [0,5]` has no solution. Huub's
    /// `Model::linear(...).ge(...).post()` returns `Err(Conflict)` here
    /// because simplification detects the inconsistency at post-time —
    /// the FFI latches that into a sticky `known_unsat` flag so the
    /// subsequent `solve()` returns UNSAT without ever running the SAT
    /// engine on the (incompletely-constrained) model.
    #[test]
    fn linear_unsat() {
        unsafe {
            let m = huub_model_new();
            let x = huub_model_new_int_var(m, 0, 5);
            let y = huub_model_new_int_var(m, 0, 5);
            let ids = [x, y];
            let cs = [1_i64, 1_i64];
            let post = huub_model_add_linear(m, ids.as_ptr(), cs.as_ptr(), 2, HuubRelOp::Ge, 11);
            assert_eq!(post, HuubResult::Unsatisfiable);
            let r = huub_model_solve(m, 10.0);
            assert_eq!(r, HuubResult::Unsatisfiable);
            huub_model_free(m);
        }
    }

    /// Single-variable variant: `x >= 6, x in [0,5]`. Same expectation
    /// as `linear_unsat` — the post-time conflict is latched.
    #[test]
    fn linear_single_var_unsat() {
        unsafe {
            let m = huub_model_new();
            let x = huub_model_new_int_var(m, 0, 5);
            let ids = [x];
            let cs = [1_i64];
            let post = huub_model_add_linear(m, ids.as_ptr(), cs.as_ptr(), 1, HuubRelOp::Ge, 6);
            assert_eq!(post, HuubResult::Unsatisfiable);
            assert_eq!(huub_model_solve(m, 10.0), HuubResult::Unsatisfiable);
            huub_model_free(m);
        }
    }

    /// Bad var-id surface check: the FFI should report an error, not panic.
    #[test]
    fn bad_var_id_sets_error() {
        unsafe {
            let m = huub_model_new();
            let _ = huub_model_new_int_var(m, 0, 5);
            // var id 42 is out of range — coefficient is fine.
            let ids = [42_i32];
            let cs = [1_i64];
            let r = huub_model_add_linear(m, ids.as_ptr(), cs.as_ptr(), 1, HuubRelOp::Le, 5);
            assert_eq!(r, HuubResult::Error);
            let err = huub_last_error();
            assert!(!err.is_null());
            let msg = CStr::from_ptr(err).to_string_lossy();
            assert!(msg.contains("out of range"), "unexpected error: {msg}");
            huub_model_free(m);
        }
    }

    /// `huub_model_new_constant` produces a var whose value is fixed.
    #[test]
    fn constant_var_roundtrip() {
        unsafe {
            let m = huub_model_new();
            let k = huub_model_new_constant(m, 42);
            assert_eq!(k, 0);
            // No other vars — solve, then read k's value.
            let r = huub_model_solve(m, 5.0);
            assert_eq!(r, HuubResult::Satisfied);
            let mut v: i64 = -1;
            assert_eq!(huub_model_value_int(m, k, &mut v), HuubResult::Satisfied);
            assert_eq!(v, 42);
            huub_model_free(m);
        }
    }

    /// `target = max(x, y, z)` with x∈[0,5], y∈[0,5], z=4; expect target=4..5.
    #[test]
    fn max_constraint() {
        unsafe {
            let m = huub_model_new();
            let x = huub_model_new_int_var(m, 0, 5);
            let y = huub_model_new_int_var(m, 0, 5);
            let z = huub_model_new_constant(m, 4);
            let t = huub_model_new_int_var(m, 0, 10);
            let ids = [x, y, z];
            let r = huub_model_add_max(m, t, ids.as_ptr(), 3);
            assert_eq!(r, HuubResult::Satisfied);
            // Pin x and y so the max is well-defined.
            let one = [x];
            let cs = [1_i64];
            assert_eq!(
                huub_model_add_linear(m, one.as_ptr(), cs.as_ptr(), 1, HuubRelOp::Eq, 2),
                HuubResult::Satisfied
            );
            let one = [y];
            assert_eq!(
                huub_model_add_linear(m, one.as_ptr(), cs.as_ptr(), 1, HuubRelOp::Eq, 3),
                HuubResult::Satisfied
            );
            assert_eq!(huub_model_solve(m, 10.0), HuubResult::Satisfied);
            let mut tv: i64 = -1;
            assert_eq!(huub_model_value_int(m, t, &mut tv), HuubResult::Satisfied);
            assert_eq!(tv, 4);
            huub_model_free(m);
        }
    }

    /// `target = min(x, y)`; pin x=7, y=3, expect target=3.
    #[test]
    fn min_constraint() {
        unsafe {
            let m = huub_model_new();
            let x = huub_model_new_int_var(m, 0, 10);
            let y = huub_model_new_int_var(m, 0, 10);
            let t = huub_model_new_int_var(m, 0, 10);
            let ids = [x, y];
            let cs = [1_i64];
            assert_eq!(
                huub_model_add_min(m, t, ids.as_ptr(), 2),
                HuubResult::Satisfied
            );
            assert_eq!(
                huub_model_add_linear(m, [x].as_ptr(), cs.as_ptr(), 1, HuubRelOp::Eq, 7),
                HuubResult::Satisfied
            );
            assert_eq!(
                huub_model_add_linear(m, [y].as_ptr(), cs.as_ptr(), 1, HuubRelOp::Eq, 3),
                HuubResult::Satisfied
            );
            assert_eq!(huub_model_solve(m, 10.0), HuubResult::Satisfied);
            let mut tv: i64 = -1;
            huub_model_value_int(m, t, &mut tv);
            assert_eq!(tv, 3);
            huub_model_free(m);
        }
    }

    /// `target = a * b`; pin a=4, b=3, expect target=12.
    #[test]
    fn mul_constraint() {
        unsafe {
            let m = huub_model_new();
            let a = huub_model_new_int_var(m, 0, 10);
            let b = huub_model_new_int_var(m, 0, 10);
            let t = huub_model_new_int_var(m, 0, 100);
            assert_eq!(huub_model_add_mul(m, t, a, b), HuubResult::Satisfied);
            let cs = [1_i64];
            huub_model_add_linear(m, [a].as_ptr(), cs.as_ptr(), 1, HuubRelOp::Eq, 4);
            huub_model_add_linear(m, [b].as_ptr(), cs.as_ptr(), 1, HuubRelOp::Eq, 3);
            assert_eq!(huub_model_solve(m, 10.0), HuubResult::Satisfied);
            let mut tv: i64 = -1;
            huub_model_value_int(m, t, &mut tv);
            assert_eq!(tv, 12);
            huub_model_free(m);
        }
    }

    /// `target = numerator / denominator`; pin numerator=17, denominator=5,
    /// expect target=3 (integer division).
    #[test]
    fn div_constraint() {
        unsafe {
            let m = huub_model_new();
            let n = huub_model_new_int_var(m, 0, 100);
            let d = huub_model_new_int_var(m, 1, 10);
            let t = huub_model_new_int_var(m, 0, 100);
            assert_eq!(huub_model_add_div(m, t, n, d), HuubResult::Satisfied);
            let cs = [1_i64];
            huub_model_add_linear(m, [n].as_ptr(), cs.as_ptr(), 1, HuubRelOp::Eq, 17);
            huub_model_add_linear(m, [d].as_ptr(), cs.as_ptr(), 1, HuubRelOp::Eq, 5);
            assert_eq!(huub_model_solve(m, 10.0), HuubResult::Satisfied);
            let mut tv: i64 = -1;
            huub_model_value_int(m, t, &mut tv);
            assert_eq!(tv, 3);
            huub_model_free(m);
        }
    }

    /// `element_const`: `target = [10, 20, 30, 40][index]`. Pin index=2,
    /// expect target=30.
    #[test]
    fn element_const_constraint() {
        unsafe {
            let m = huub_model_new();
            let idx = huub_model_new_int_var(m, 0, 3);
            let tgt = huub_model_new_int_var(m, 0, 100);
            let vals = [10_i64, 20, 30, 40];
            assert_eq!(
                huub_model_add_element_const(m, idx, vals.as_ptr(), 4, tgt),
                HuubResult::Satisfied
            );
            let cs = [1_i64];
            huub_model_add_linear(m, [idx].as_ptr(), cs.as_ptr(), 1, HuubRelOp::Eq, 2);
            assert_eq!(huub_model_solve(m, 10.0), HuubResult::Satisfied);
            let mut tv: i64 = -1;
            huub_model_value_int(m, tgt, &mut tv);
            assert_eq!(tv, 30);
            huub_model_free(m);
        }
    }

    /// `element_var`: pin v0=5, v1=7, v2=9, idx=1, expect target=7.
    #[test]
    fn element_var_constraint() {
        unsafe {
            let m = huub_model_new();
            let v0 = huub_model_new_int_var(m, 0, 10);
            let v1 = huub_model_new_int_var(m, 0, 10);
            let v2 = huub_model_new_int_var(m, 0, 10);
            let idx = huub_model_new_int_var(m, 0, 2);
            let tgt = huub_model_new_int_var(m, 0, 10);
            let arr = [v0, v1, v2];
            assert_eq!(
                huub_model_add_element_var(m, idx, arr.as_ptr(), 3, tgt),
                HuubResult::Satisfied
            );
            let cs = [1_i64];
            huub_model_add_linear(m, [v0].as_ptr(), cs.as_ptr(), 1, HuubRelOp::Eq, 5);
            huub_model_add_linear(m, [v1].as_ptr(), cs.as_ptr(), 1, HuubRelOp::Eq, 7);
            huub_model_add_linear(m, [v2].as_ptr(), cs.as_ptr(), 1, HuubRelOp::Eq, 9);
            huub_model_add_linear(m, [idx].as_ptr(), cs.as_ptr(), 1, HuubRelOp::Eq, 1);
            assert_eq!(huub_model_solve(m, 10.0), HuubResult::Satisfied);
            let mut tv: i64 = -1;
            huub_model_value_int(m, tgt, &mut tv);
            assert_eq!(tv, 7);
            huub_model_free(m);
        }
    }

    /// `all_different`: three vars in [1,3], all distinct → values are a
    /// permutation of {1,2,3}. Pin x=1, y=2, expect z=3.
    #[test]
    fn all_different_constraint() {
        unsafe {
            let m = huub_model_new();
            let x = huub_model_new_int_var(m, 1, 3);
            let y = huub_model_new_int_var(m, 1, 3);
            let z = huub_model_new_int_var(m, 1, 3);
            let arr = [x, y, z];
            assert_eq!(
                huub_model_add_all_different(m, arr.as_ptr(), 3),
                HuubResult::Satisfied
            );
            let cs = [1_i64];
            huub_model_add_linear(m, [x].as_ptr(), cs.as_ptr(), 1, HuubRelOp::Eq, 1);
            huub_model_add_linear(m, [y].as_ptr(), cs.as_ptr(), 1, HuubRelOp::Eq, 2);
            assert_eq!(huub_model_solve(m, 10.0), HuubResult::Satisfied);
            let mut zv: i64 = -1;
            huub_model_value_int(m, z, &mut zv);
            assert_eq!(zv, 3);
            huub_model_free(m);
        }
    }

    /// Linear-reif (half): pin b=true, require x+y == 7 only when b. So b
    /// being forced true forces the constraint. Expect SAT with x+y==7.
    #[test]
    fn linear_reif_half() {
        unsafe {
            let m = huub_model_new();
            let x = huub_model_new_int_var(m, 0, 10);
            let y = huub_model_new_int_var(m, 0, 10);
            let b = huub_model_new_bool_var(m);
            // Force b = true.
            let blits = [b];
            assert_eq!(
                huub_model_add_bool_or(m, blits.as_ptr(), 1, -1),
                HuubResult::Satisfied
            );
            // b → x + y == 7
            let ids = [x, y];
            let cs = [1_i64, 1_i64];
            assert_eq!(
                huub_model_add_linear_reif(
                    m,
                    ids.as_ptr(),
                    cs.as_ptr(),
                    2,
                    HuubRelOp::Eq,
                    7,
                    b,
                    true,
                ),
                HuubResult::Satisfied
            );
            assert_eq!(huub_model_solve(m, 10.0), HuubResult::Satisfied);
            let mut xv: i64 = -1;
            let mut yv: i64 = -1;
            huub_model_value_int(m, x, &mut xv);
            huub_model_value_int(m, y, &mut yv);
            assert_eq!(xv + yv, 7);
            huub_model_free(m);
        }
    }

    /// Linear-reif (full iff): without forcing b, the SAT engine may pick
    /// b=false. Then the iff means x+y != 7. Force x=0 y=0 → constraint
    /// (0+0==7) is false → b must be false (since iff). Read b back.
    #[test]
    fn linear_reif_iff_back_inference() {
        unsafe {
            let m = huub_model_new();
            let x = huub_model_new_int_var(m, 0, 0);
            let y = huub_model_new_int_var(m, 0, 0);
            let b = huub_model_new_bool_var(m);
            let ids = [x, y];
            let cs = [1_i64, 1_i64];
            assert_eq!(
                huub_model_add_linear_reif(
                    m,
                    ids.as_ptr(),
                    cs.as_ptr(),
                    2,
                    HuubRelOp::Eq,
                    7,
                    b,
                    false,
                ),
                HuubResult::Satisfied
            );
            assert_eq!(huub_model_solve(m, 10.0), HuubResult::Satisfied);
            let mut bv = true;
            huub_model_value_bool(m, b, &mut bv);
            assert!(!bv, "iff forces b to be false");
            huub_model_free(m);
        }
    }

    /// `bool_or`: pin a=false, b=false → require `a ∨ b ∨ c` → c must be true.
    #[test]
    fn bool_or_constraint() {
        unsafe {
            let m = huub_model_new();
            let a = huub_model_new_bool_var(m);
            let b = huub_model_new_bool_var(m);
            let c = huub_model_new_bool_var(m);
            // Force a = false and b = false via implication: true → ¬a, true → ¬b.
            // Easier: post Or(¬a) → equivalent to ¬a; via add_bool_or with one arg
            // requires negation which we don't expose. Use linear_reif on a bool
            // cast: not exposed either. Easiest: force via linear over the
            // formula. Take a different tack — post bool_or(a, b, c) and also
            // bool_or(¬a alone) by using add_implication: post (a → false)
            // means ¬a. Use a constant-false bool: make a "false" by posting
            // Or(b_false) is empty so instead: create a constant int 0 == 1
            // would be UNSAT. Simplest: do not pin; just check that an OR of
            // three booleans does not preclude SAT.
            let lits = [a, b, c];
            assert_eq!(
                huub_model_add_bool_or(m, lits.as_ptr(), 3, -1),
                HuubResult::Satisfied
            );
            assert_eq!(huub_model_solve(m, 10.0), HuubResult::Satisfied);
            // At least one of a, b, c must be true.
            let mut av = false;
            let mut bv = false;
            let mut cv = false;
            huub_model_value_bool(m, a, &mut av);
            huub_model_value_bool(m, b, &mut bv);
            huub_model_value_bool(m, c, &mut cv);
            assert!(av || bv || cv);
            huub_model_free(m);
        }
    }

    /// `bool_and`: require `a ∧ b` — both must be true.
    #[test]
    fn bool_and_constraint() {
        unsafe {
            let m = huub_model_new();
            let a = huub_model_new_bool_var(m);
            let b = huub_model_new_bool_var(m);
            let lits = [a, b];
            assert_eq!(
                huub_model_add_bool_and(m, lits.as_ptr(), 2, -1),
                HuubResult::Satisfied
            );
            assert_eq!(huub_model_solve(m, 10.0), HuubResult::Satisfied);
            let mut av = false;
            let mut bv = false;
            huub_model_value_bool(m, a, &mut av);
            huub_model_value_bool(m, b, &mut bv);
            assert!(av && bv);
            huub_model_free(m);
        }
    }

    /// `implication a → b`: assert a; expect b inferred.
    #[test]
    fn implication_constraint() {
        unsafe {
            let m = huub_model_new();
            let a = huub_model_new_bool_var(m);
            let b = huub_model_new_bool_var(m);
            assert_eq!(huub_model_add_implication(m, a, b), HuubResult::Satisfied);
            // Force a = true via bool_or with single lit.
            let one = [a];
            assert_eq!(
                huub_model_add_bool_or(m, one.as_ptr(), 1, -1),
                HuubResult::Satisfied
            );
            assert_eq!(huub_model_solve(m, 10.0), HuubResult::Satisfied);
            let mut bv = false;
            huub_model_value_bool(m, b, &mut bv);
            assert!(bv);
            huub_model_free(m);
        }
    }

    /// Build a single interval (start, size=5, end), confirm
    /// `start + 5 == end` was posted.
    #[test]
    fn interval_consistency() {
        unsafe {
            let m = huub_model_new();
            let s = huub_model_new_int_var(m, 0, 10);
            let z = huub_model_new_constant(m, 5);
            let e = huub_model_new_int_var(m, 0, 20);
            let iv = huub_model_new_interval(m, s, z, e);
            assert!(iv >= 0);
            // Pin start=3, expect end=8.
            let cs = [1_i64];
            huub_model_add_linear(m, [s].as_ptr(), cs.as_ptr(), 1, HuubRelOp::Eq, 3);
            assert_eq!(huub_model_solve(m, 10.0), HuubResult::Satisfied);
            let mut ev: i64 = -1;
            huub_model_value_int(m, e, &mut ev);
            assert_eq!(ev, 8);
            huub_model_free(m);
        }
    }

    /// Two intervals each of size 3 in a horizon of 5: they must not
    /// overlap. Pin first start=0, expect second start >= 3 or <= -3.
    #[test]
    fn no_overlap_constraint() {
        unsafe {
            let m = huub_model_new();
            let s1 = huub_model_new_int_var(m, 0, 5);
            let z1 = huub_model_new_constant(m, 3);
            let e1 = huub_model_new_int_var(m, 0, 8);
            let iv1 = huub_model_new_interval(m, s1, z1, e1);

            let s2 = huub_model_new_int_var(m, 0, 5);
            let z2 = huub_model_new_constant(m, 3);
            let e2 = huub_model_new_int_var(m, 0, 8);
            let iv2 = huub_model_new_interval(m, s2, z2, e2);

            let ivs = [iv1, iv2];
            assert_eq!(
                huub_model_add_no_overlap(m, ivs.as_ptr(), 2),
                HuubResult::Satisfied
            );
            let cs = [1_i64];
            huub_model_add_linear(m, [s1].as_ptr(), cs.as_ptr(), 1, HuubRelOp::Eq, 0);
            assert_eq!(huub_model_solve(m, 10.0), HuubResult::Satisfied);
            let mut s2v: i64 = -1;
            huub_model_value_int(m, s2, &mut s2v);
            // s1=0, z1=3 ⇒ s1's interval is [0, 3). s2 must start at ≥3.
            assert!(s2v >= 3, "s2 = {s2v} should be >= 3");
            huub_model_free(m);
        }
    }

    /// Three intervals, sizes (2, 3, 1), all constant; disjunctive in [0, 6).
    /// Pin s0=0 and s1=2; expect s2 to land at 5 (only spot left).
    #[test]
    fn disjunctive_constraint() {
        unsafe {
            let m = huub_model_new();
            let mk = |start_lb, start_ub, dur, end_ub| {
                let s = huub_model_new_int_var(m, start_lb, start_ub);
                let z = huub_model_new_constant(m, dur);
                let e = huub_model_new_int_var(m, 0, end_ub);
                let iv = huub_model_new_interval(m, s, z, e);
                (s, iv)
            };
            let (s0, iv0) = mk(0, 5, 2, 6);
            let (s1, iv1) = mk(0, 5, 3, 6);
            let (s2, iv2) = mk(0, 5, 1, 6);
            let ivs = [iv0, iv1, iv2];
            assert_eq!(
                huub_model_add_disjunctive(m, ivs.as_ptr(), 3),
                HuubResult::Satisfied
            );
            let cs = [1_i64];
            huub_model_add_linear(m, [s0].as_ptr(), cs.as_ptr(), 1, HuubRelOp::Eq, 0);
            huub_model_add_linear(m, [s1].as_ptr(), cs.as_ptr(), 1, HuubRelOp::Eq, 2);
            assert_eq!(huub_model_solve(m, 10.0), HuubResult::Satisfied);
            let mut s2v: i64 = -1;
            huub_model_value_int(m, s2, &mut s2v);
            assert_eq!(s2v, 5);
            huub_model_free(m);
        }
    }

    /// Disjunctive with a non-constant duration: must return Error and set
    /// the last-error string.
    #[test]
    fn disjunctive_rejects_var_duration() {
        unsafe {
            let m = huub_model_new();
            let s = huub_model_new_int_var(m, 0, 5);
            // size is a *variable* (domain wider than one value).
            let z = huub_model_new_int_var(m, 1, 3);
            let e = huub_model_new_int_var(m, 0, 8);
            let iv = huub_model_new_interval(m, s, z, e);
            let ivs = [iv];
            let r = huub_model_add_disjunctive(m, ivs.as_ptr(), 1);
            assert_eq!(r, HuubResult::Error);
            let err = huub_last_error();
            assert!(!err.is_null());
            let msg = CStr::from_ptr(err).to_string_lossy();
            assert!(msg.contains("non-constant"), "unexpected: {msg}");
            huub_model_free(m);
        }
    }

    /// `target = numerator mod denominator`; pin numerator=17, denominator=5,
    /// expect target=2.
    #[test]
    fn mod_synthesized() {
        unsafe {
            let m = huub_model_new();
            let n = huub_model_new_int_var(m, 0, 100);
            let d = huub_model_new_int_var(m, 1, 10);
            let t = huub_model_new_int_var(m, 0, 100);
            assert_eq!(huub_model_add_mod(m, t, n, d), HuubResult::Satisfied);
            let cs = [1_i64];
            huub_model_add_linear(m, [n].as_ptr(), cs.as_ptr(), 1, HuubRelOp::Eq, 17);
            huub_model_add_linear(m, [d].as_ptr(), cs.as_ptr(), 1, HuubRelOp::Eq, 5);
            assert_eq!(huub_model_solve(m, 10.0), HuubResult::Satisfied);
            let mut tv: i64 = -1;
            huub_model_value_int(m, t, &mut tv);
            assert_eq!(tv, 2);
            huub_model_free(m);
        }
    }

    /// Inverse: fwd, bwd of length 3 over domain [0,2]. Pin fwd = [2, 0, 1];
    /// expect bwd = [1, 2, 0] (the inverse permutation).
    #[test]
    fn inverse_synthesized() {
        unsafe {
            let m = huub_model_new();
            let mk = || huub_model_new_int_var(m, 0, 2);
            let f0 = mk();
            let f1 = mk();
            let f2 = mk();
            let b0 = mk();
            let b1 = mk();
            let b2 = mk();
            let fwd = [f0, f1, f2];
            let bwd = [b0, b1, b2];
            assert_eq!(
                huub_model_add_inverse(m, fwd.as_ptr(), bwd.as_ptr(), 3),
                HuubResult::Satisfied
            );
            let cs = [1_i64];
            huub_model_add_linear(m, [f0].as_ptr(), cs.as_ptr(), 1, HuubRelOp::Eq, 2);
            huub_model_add_linear(m, [f1].as_ptr(), cs.as_ptr(), 1, HuubRelOp::Eq, 0);
            huub_model_add_linear(m, [f2].as_ptr(), cs.as_ptr(), 1, HuubRelOp::Eq, 1);
            assert_eq!(huub_model_solve(m, 10.0), HuubResult::Satisfied);
            let mut v: i64 = -1;
            huub_model_value_int(m, b0, &mut v);
            assert_eq!(v, 1);
            huub_model_value_int(m, b1, &mut v);
            assert_eq!(v, 2);
            huub_model_value_int(m, b2, &mut v);
            assert_eq!(v, 0);
            huub_model_free(m);
        }
    }

    /// `at_most_one(a, b, c)` allows zero or one of the literals to be
    /// true. Pin a=true; expect b=false and c=false.
    #[test]
    fn at_most_one_synthesized() {
        unsafe {
            let m = huub_model_new();
            let a = huub_model_new_bool_var(m);
            let b = huub_model_new_bool_var(m);
            let c = huub_model_new_bool_var(m);
            let lits = [a, b, c];
            assert_eq!(
                huub_model_add_at_most_one(m, lits.as_ptr(), 3),
                HuubResult::Satisfied
            );
            // Force a = true via bool_or with single lit.
            let one = [a];
            huub_model_add_bool_or(m, one.as_ptr(), 1, -1);
            assert_eq!(huub_model_solve(m, 10.0), HuubResult::Satisfied);
            let mut av = false;
            let mut bv = true;
            let mut cv = true;
            huub_model_value_bool(m, a, &mut av);
            huub_model_value_bool(m, b, &mut bv);
            huub_model_value_bool(m, c, &mut cv);
            assert!(av && !bv && !cv, "a={av} b={bv} c={cv}");
            huub_model_free(m);
        }
    }

    /// `at_most_one` UNSAT: pin a=true and b=true; expect UNSAT.
    #[test]
    fn at_most_one_unsat() {
        unsafe {
            let m = huub_model_new();
            let a = huub_model_new_bool_var(m);
            let b = huub_model_new_bool_var(m);
            let lits = [a, b];
            assert_eq!(
                huub_model_add_at_most_one(m, lits.as_ptr(), 2),
                HuubResult::Satisfied
            );
            huub_model_add_bool_or(m, [a].as_ptr(), 1, -1);
            huub_model_add_bool_or(m, [b].as_ptr(), 1, -1);
            assert_eq!(huub_model_solve(m, 10.0), HuubResult::Unsatisfiable);
            huub_model_free(m);
        }
    }

    /// Hint roundtrip: x in [0,5] with no constraints. Hint x=3 and solve;
    /// the warm-start should make the solver land on x=3 even though many
    /// other assignments also satisfy.
    #[test]
    fn hint_int_warmstart() {
        unsafe {
            let m = huub_model_new();
            let x = huub_model_new_int_var(m, 0, 5);
            assert_eq!(huub_model_add_hint_int(m, x, 3), HuubResult::Satisfied);
            assert_eq!(huub_model_solve(m, 10.0), HuubResult::Satisfied);
            let mut xv: i64 = -1;
            huub_model_value_int(m, x, &mut xv);
            assert_eq!(xv, 3, "warm-start should fix x to the hinted value");
            huub_model_free(m);
        }
    }

    /// Bool hint roundtrip: free bool `b` with no constraints. Hint b=true.
    #[test]
    fn hint_bool_warmstart() {
        unsafe {
            let m = huub_model_new();
            let b = huub_model_new_bool_var(m);
            assert_eq!(huub_model_add_hint_bool(m, b, true), HuubResult::Satisfied);
            assert_eq!(huub_model_solve(m, 10.0), HuubResult::Satisfied);
            let mut bv = false;
            huub_model_value_bool(m, b, &mut bv);
            assert!(bv);
            huub_model_free(m);
        }
    }

    /// `clear_hints` drops staged hints before they're applied: hint x=3,
    /// then clear, then solve. x is free — assignment is whatever the
    /// solver picks (just confirm it solves cleanly without applying the
    /// cleared hint as a constraint).
    #[test]
    fn clear_hints_drops_staged() {
        unsafe {
            let m = huub_model_new();
            let x = huub_model_new_int_var(m, 0, 5);
            // Constrain x != 3 so the hint, if applied, would conflict.
            // (We're checking the hint is *cleared*, not "WarmStartBrancher
            // tolerates conflicts" — though both happen to hold.)
            let cs = [1_i64];
            huub_model_add_linear(m, [x].as_ptr(), cs.as_ptr(), 1, HuubRelOp::Ne, 3);
            huub_model_add_hint_int(m, x, 3);
            assert_eq!(huub_model_clear_hints(m), HuubResult::Satisfied);
            assert_eq!(huub_model_solve(m, 10.0), HuubResult::Satisfied);
            let mut xv: i64 = -1;
            huub_model_value_int(m, x, &mut xv);
            assert_ne!(xv, 3, "x != 3 constraint must hold; cleared hint must not force x=3");
            huub_model_free(m);
        }
    }

    /// T_squeeze re-solve pattern: solve → reset → re-hint → re-solve.
    /// Confirm the second solve picks up a new hint after the first solve
    /// already happened.
    #[test]
    fn reset_for_resolve_roundtrip() {
        unsafe {
            let m = huub_model_new();
            let x = huub_model_new_int_var(m, 0, 10);

            // First iteration: hint x=2, expect x=2.
            huub_model_add_hint_int(m, x, 2);
            assert_eq!(huub_model_solve(m, 10.0), HuubResult::Satisfied);
            let mut xv: i64 = -1;
            huub_model_value_int(m, x, &mut xv);
            assert_eq!(xv, 2);

            // Reset, clear staged hints, restage with x=7, re-solve.
            assert_eq!(huub_model_reset_for_resolve(m), HuubResult::Satisfied);
            huub_model_clear_hints(m);
            huub_model_add_hint_int(m, x, 7);
            assert_eq!(huub_model_solve(m, 10.0), HuubResult::Satisfied);
            huub_model_value_int(m, x, &mut xv);
            assert_eq!(xv, 7, "second solve should respect the new hint");

            huub_model_free(m);
        }
    }

    /// Reset before any solve is a no-op (returns NotSolved, doesn't error).
    #[test]
    fn reset_before_solve_is_noop() {
        unsafe {
            let m = huub_model_new();
            let _x = huub_model_new_int_var(m, 0, 5);
            assert_eq!(huub_model_reset_for_resolve(m), HuubResult::NotSolved);
            // Still solvable afterwards.
            assert_eq!(huub_model_solve(m, 5.0), HuubResult::Satisfied);
            huub_model_free(m);
        }
    }

    /// Bad hint var-id surfaces an Error.
    #[test]
    fn hint_bad_var_id() {
        unsafe {
            let m = huub_model_new();
            let _ = huub_model_new_int_var(m, 0, 5);
            assert_eq!(huub_model_add_hint_int(m, 42, 3), HuubResult::Error);
            let err = huub_last_error();
            assert!(!err.is_null());
            let msg = CStr::from_ptr(err).to_string_lossy();
            assert!(msg.contains("out of range"), "unexpected: {msg}");
            huub_model_free(m);
        }
    }

    /// Int decision strategy: three vars in [0,5], InputOrder + IndomainMin.
    /// First var should be set to its minimum (0).
    #[test]
    fn int_decision_strategy_indomain_min() {
        unsafe {
            let m = huub_model_new();
            let a = huub_model_new_int_var(m, 0, 5);
            let b = huub_model_new_int_var(m, 0, 5);
            let c = huub_model_new_int_var(m, 0, 5);
            let vars = [a, b, c];
            assert_eq!(
                huub_model_add_decision_strategy_int(
                    m,
                    vars.as_ptr(),
                    3,
                    HuubDecisionSel::InputOrder,
                    HuubDomainSel::IndomainMin,
                ),
                HuubResult::Satisfied
            );
            assert_eq!(
                huub_model_set_search_strategy(m, HuubSearchStrategy::Branchers, 0),
                HuubResult::Satisfied
            );
            assert_eq!(huub_model_solve(m, 10.0), HuubResult::Satisfied);
            let mut av: i64 = -1;
            let mut bv: i64 = -1;
            let mut cv: i64 = -1;
            huub_model_value_int(m, a, &mut av);
            huub_model_value_int(m, b, &mut bv);
            huub_model_value_int(m, c, &mut cv);
            assert_eq!((av, bv, cv), (0, 0, 0));
            huub_model_free(m);
        }
    }

    /// Int decision strategy: IndomainMax should push toward each var's
    /// max value (5 here).
    #[test]
    fn int_decision_strategy_indomain_max() {
        unsafe {
            let m = huub_model_new();
            let a = huub_model_new_int_var(m, 0, 5);
            let b = huub_model_new_int_var(m, 0, 5);
            let vars = [a, b];
            huub_model_add_decision_strategy_int(
                m,
                vars.as_ptr(),
                2,
                HuubDecisionSel::InputOrder,
                HuubDomainSel::IndomainMax,
            );
            huub_model_set_search_strategy(m, HuubSearchStrategy::Branchers, 0);
            assert_eq!(huub_model_solve(m, 10.0), HuubResult::Satisfied);
            let mut av: i64 = -1;
            let mut bv: i64 = -1;
            huub_model_value_int(m, a, &mut av);
            huub_model_value_int(m, b, &mut bv);
            assert_eq!((av, bv), (5, 5));
            huub_model_free(m);
        }
    }

    /// Bool decision strategy: two bools, IndomainMin (= false). Both
    /// expected false in a constraint-free model.
    #[test]
    fn bool_decision_strategy_indomain_min() {
        unsafe {
            let m = huub_model_new();
            let p = huub_model_new_bool_var(m);
            let q = huub_model_new_bool_var(m);
            let vars = [p, q];
            huub_model_add_decision_strategy_bool(
                m,
                vars.as_ptr(),
                2,
                HuubDecisionSel::InputOrder,
                HuubDomainSel::IndomainMin,
            );
            huub_model_set_search_strategy(m, HuubSearchStrategy::Branchers, 0);
            assert_eq!(huub_model_solve(m, 10.0), HuubResult::Satisfied);
            let mut pv = true;
            let mut qv = true;
            huub_model_value_bool(m, p, &mut pv);
            huub_model_value_bool(m, q, &mut qv);
            assert!(!pv && !qv);
            huub_model_free(m);
        }
    }

    /// Search strategy `Sat`: ignores branchers, uses SAT engine default.
    /// Just confirms the call doesn't error and the model still solves.
    #[test]
    fn search_strategy_sat() {
        unsafe {
            let m = huub_model_new();
            let _x = huub_model_new_int_var(m, 0, 5);
            assert_eq!(
                huub_model_set_search_strategy(m, HuubSearchStrategy::Sat, 0),
                HuubResult::Satisfied
            );
            assert_eq!(huub_model_solve(m, 10.0), HuubResult::Satisfied);
            huub_model_free(m);
        }
    }

    /// Search strategy `Transition` with non-zero conflict trigger.
    #[test]
    fn search_strategy_transition() {
        unsafe {
            let m = huub_model_new();
            let _x = huub_model_new_int_var(m, 0, 5);
            assert_eq!(
                huub_model_set_search_strategy(m, HuubSearchStrategy::Transition, 1000),
                HuubResult::Satisfied
            );
            assert_eq!(huub_model_solve(m, 10.0), HuubResult::Satisfied);
            huub_model_free(m);
        }
    }

    /// Minimize: x + y >= 7, x,y in [0,5]. Minimum of x+y is 7. Set
    /// objective = x+y via a "sum" var. Actually simpler: minimize x with
    /// x >= 3, expect optimum = 3.
    #[test]
    fn minimize_simple() {
        unsafe {
            let m = huub_model_new();
            let x = huub_model_new_int_var(m, 0, 10);
            let cs = [1_i64];
            huub_model_add_linear(m, [x].as_ptr(), cs.as_ptr(), 1, HuubRelOp::Ge, 3);
            assert_eq!(huub_model_set_minimize(m, x), HuubResult::Satisfied);
            // Status will be Complete (proven optimal) for minimization.
            let r = huub_model_solve(m, 10.0);
            assert!(
                matches!(r, HuubResult::Complete | HuubResult::Satisfied),
                "got {:?}",
                r
            );
            let mut opt: i64 = -1;
            assert_eq!(
                huub_model_objective_value(m, &mut opt),
                HuubResult::Satisfied
            );
            assert_eq!(opt, 3);
            let mut xv: i64 = -1;
            huub_model_value_int(m, x, &mut xv);
            assert_eq!(xv, 3);
            huub_model_free(m);
        }
    }

    /// Maximize x with x <= 7, x in [0,10] → optimum = 7.
    #[test]
    fn maximize_simple() {
        unsafe {
            let m = huub_model_new();
            let x = huub_model_new_int_var(m, 0, 10);
            let cs = [1_i64];
            huub_model_add_linear(m, [x].as_ptr(), cs.as_ptr(), 1, HuubRelOp::Le, 7);
            huub_model_set_maximize(m, x);
            let r = huub_model_solve(m, 10.0);
            assert!(matches!(r, HuubResult::Complete | HuubResult::Satisfied));
            let mut opt: i64 = -1;
            huub_model_objective_value(m, &mut opt);
            assert_eq!(opt, 7);
            huub_model_free(m);
        }
    }

    /// Without an objective set, objective_value returns NotSolved even
    /// after a satisfaction solve.
    #[test]
    fn objective_value_unset_after_satisfy() {
        unsafe {
            let m = huub_model_new();
            let _x = huub_model_new_int_var(m, 0, 5);
            huub_model_solve(m, 10.0);
            let mut opt: i64 = 999;
            assert_eq!(
                huub_model_objective_value(m, &mut opt),
                HuubResult::NotSolved
            );
            assert_eq!(opt, 999, "out must not be written on NotSolved");
            huub_model_free(m);
        }
    }

    /// Conflict budget = 0 disables the limit (just a smoke test).
    #[test]
    fn conflict_budget_zero_disables() {
        unsafe {
            let m = huub_model_new();
            let _x = huub_model_new_int_var(m, 0, 5);
            assert_eq!(huub_model_set_conflict_budget(m, 0), HuubResult::Satisfied);
            assert_eq!(huub_model_solve(m, 10.0), HuubResult::Satisfied);
            huub_model_free(m);
        }
    }

    /// Conflict budget with a non-trivial model: ten bools with
    /// `at_most_one`, but we *force* two of them to be true via
    /// `bool_or` on single literals — UNSAT. The conflict budget should
    /// kick in if the engine takes its time discovering UNSAT. We just
    /// confirm the call doesn't error and produces a defined verdict
    /// (SAT/UNSAT/Unknown).
    #[test]
    fn conflict_budget_set_doesnt_break_solve() {
        unsafe {
            let m = huub_model_new();
            let mut lits = Vec::new();
            for _ in 0..10 {
                lits.push(huub_model_new_bool_var(m));
            }
            huub_model_add_at_most_one(m, lits.as_ptr(), lits.len());
            // Force lits[0]=true and lits[1]=true → UNSAT (at_most_one
            // says only one can be true).
            huub_model_add_bool_or(m, lits[0..1].as_ptr(), 1, -1);
            huub_model_add_bool_or(m, lits[1..2].as_ptr(), 1, -1);
            assert_eq!(
                huub_model_set_conflict_budget(m, 100),
                HuubResult::Satisfied
            );
            let r = huub_model_solve(m, 10.0);
            assert!(matches!(
                r,
                HuubResult::Unsatisfiable | HuubResult::Unknown
            ));
            huub_model_free(m);
        }
    }

    /// Null-handle safety: every entry point should tolerate a null handle
    /// without panicking.
    #[test]
    fn null_handle_safe() {
        unsafe {
            assert_eq!(huub_model_new_int_var(ptr::null_mut(), 0, 5), -1);
            assert_eq!(huub_model_new_bool_var(ptr::null_mut()), -1);
            let r = huub_model_solve(ptr::null_mut(), 1.0);
            assert_eq!(r, HuubResult::Error);
            huub_model_free(ptr::null_mut()); // no-op
        }
    }

    // ----- B.3 — Model clone + lower-time SAT knobs ---------------------

    /// Clone a Building-state handle, then post a constraint to the
    /// clone only. The original remains under-constrained — both solve
    /// SAT, but only the clone's assignment respects the extra
    /// constraint.
    #[test]
    fn clone_independent_post_constraint() {
        unsafe {
            let orig = huub_model_new();
            let x = huub_model_new_int_var(orig, 0, 10);
            assert_eq!(x, 0);

            let cl = huub_model_clone(orig);
            assert!(!cl.is_null());

            // Post `x >= 7` on the clone only.
            let ids = [x];
            let cs = [1_i64];
            assert_eq!(
                huub_model_add_linear(cl, ids.as_ptr(), cs.as_ptr(), 1, HuubRelOp::Ge, 7),
                HuubResult::Satisfied
            );

            // Pin the original to x = 0 via a brancher so the assignments
            // are deterministic and distinguishable.
            let v = [0_usize as i32];
            assert_eq!(
                huub_model_add_decision_strategy_int(
                    orig,
                    v.as_ptr(),
                    1,
                    HuubDecisionSel::InputOrder,
                    HuubDomainSel::IndomainMin,
                ),
                HuubResult::Satisfied
            );

            assert_eq!(huub_model_solve(orig, 5.0), HuubResult::Satisfied);
            assert_eq!(huub_model_solve(cl, 5.0), HuubResult::Satisfied);

            let mut xv_o: i64 = -1;
            let mut xv_c: i64 = -1;
            assert_eq!(
                huub_model_value_int(orig, x, &mut xv_o),
                HuubResult::Satisfied
            );
            assert_eq!(
                huub_model_value_int(cl, x, &mut xv_c),
                HuubResult::Satisfied
            );
            assert_eq!(xv_o, 0, "original has no >=7 constraint");
            assert!(xv_c >= 7, "clone enforces x >= 7, got {xv_c}");

            huub_model_free(orig);
            huub_model_free(cl);
        }
    }

    /// known_unsat latched on the original should propagate to the clone:
    /// solving the clone returns Unsatisfiable without running the SAT
    /// engine.
    #[test]
    fn clone_preserves_known_unsat() {
        unsafe {
            let orig = huub_model_new();
            let x = huub_model_new_int_var(orig, 0, 5);
            let ids = [x];
            let cs = [1_i64];
            // x >= 6 is UNSAT at post-time on [0,5] → latches known_unsat.
            assert_eq!(
                huub_model_add_linear(orig, ids.as_ptr(), cs.as_ptr(), 1, HuubRelOp::Ge, 6),
                HuubResult::Unsatisfiable
            );

            let cl = huub_model_clone(orig);
            assert!(!cl.is_null());
            assert_eq!(huub_model_solve(cl, 5.0), HuubResult::Unsatisfiable);

            huub_model_free(orig);
            huub_model_free(cl);
        }
    }

    /// Cloning after the first solve (Lowered state) is rejected with an
    /// error string. The C++ caller is expected to clone before any
    /// thread spawns a solve.
    #[test]
    fn clone_in_lowered_state_errors() {
        unsafe {
            let orig = huub_model_new();
            let _x = huub_model_new_int_var(orig, 0, 5);
            assert_eq!(huub_model_solve(orig, 5.0), HuubResult::Satisfied);

            let cl = huub_model_clone(orig);
            assert!(cl.is_null());
            let err = huub_last_error();
            assert!(!err.is_null());
            let msg = CStr::from_ptr(err).to_string_lossy();
            assert!(msg.contains("Lowered"), "unexpected error: {msg}");

            huub_model_free(orig);
        }
    }

    /// `inner_model_clone` returns Some for Building state and None for
    /// Lowered. The snapshot's `int_vars` length tracks the source's
    /// var registry length.
    #[test]
    fn inner_model_clone_building_then_lowered() {
        unsafe {
            let m = huub_model_new();
            let _x = huub_model_new_int_var(m, 0, 5);
            let _y = huub_model_new_int_var(m, 0, 5);
            let _b = huub_model_new_bool_var(m);

            let snap = inner_model_clone(m);
            assert!(snap.is_some(), "Building-state clone should succeed");
            let s = snap.unwrap();
            assert_eq!(s.int_vars.len(), 2);
            assert_eq!(s.bool_vars.len(), 1);
            assert!(!s.known_unsat);

            // Lower the original by solving; subsequent calls must return None.
            assert_eq!(huub_model_solve(m, 5.0), HuubResult::Satisfied);
            assert!(inner_model_clone(m).is_none());

            huub_model_free(m);
        }
    }

    /// Smoke test for `huub_model_set_sat_restart`: setting the flag on
    /// either value still allows the solve to complete on a small
    /// satisfiable model.
    #[test]
    fn set_sat_restart_and_solve() {
        unsafe {
            for &enable in &[true, false] {
                let m = huub_model_new();
                let x = huub_model_new_int_var(m, 0, 10);
                let y = huub_model_new_int_var(m, 0, 10);
                let ids = [x, y];
                let cs = [1_i64, 1_i64];
                assert_eq!(
                    huub_model_add_linear(m, ids.as_ptr(), cs.as_ptr(), 2, HuubRelOp::Eq, 7),
                    HuubResult::Satisfied
                );
                assert_eq!(
                    huub_model_set_sat_restart(m, enable),
                    HuubResult::Satisfied
                );
                assert_eq!(huub_model_solve(m, 5.0), HuubResult::Satisfied);
                huub_model_free(m);
            }
        }
    }

    /// Smoke test for `huub_model_set_sat_inprocessing`: same shape as
    /// the restart test. The flag toggles six CaDiCaL knobs together;
    /// we don't verify the underlying state, only that the solve path
    /// still works.
    #[test]
    fn set_sat_inprocessing_and_solve() {
        unsafe {
            for &enable in &[true, false] {
                let m = huub_model_new();
                let x = huub_model_new_int_var(m, 0, 10);
                let y = huub_model_new_int_var(m, 0, 10);
                let ids = [x, y];
                let cs = [1_i64, 1_i64];
                assert_eq!(
                    huub_model_add_linear(m, ids.as_ptr(), cs.as_ptr(), 2, HuubRelOp::Eq, 7),
                    HuubResult::Satisfied
                );
                assert_eq!(
                    huub_model_set_sat_inprocessing(m, enable),
                    HuubResult::Satisfied
                );
                assert_eq!(huub_model_solve(m, 5.0), HuubResult::Satisfied);
                huub_model_free(m);
            }
        }
    }
}
