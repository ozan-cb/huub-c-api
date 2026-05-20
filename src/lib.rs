//! C ABI wrapper around Huub, sized for the `ScheduleInf` productization
//! plan (see `../../PRODUCTIZATION.md`).
//!
//! v0 scope (this file): the minimum surface that exercises the full
//! pipeline end-to-end — opaque model handle, int/bool var creation, one
//! linear-constraint primitive, single-worker solve, value extraction,
//! thread-local last-error. v1+ will fill in the rest of the constraint
//! classes from `PRODUCTIZATION.md` §2 (no_overlap/disjunctive, element,
//! cumulative, all_different, max/min/mul/div/mod, reification,
//! branchers, portfolio).
//!
//! Conventions:
//!
//! * Every `extern "C" fn` wraps its body in `std::panic::catch_unwind`
//!   so a Rust panic never unwinds into a C++ frame. On panic we record
//!   the message in thread-local last-error and return an error code (or
//!   a sentinel handle).
//! * Handles are opaque (`*mut HuubModel`). All allocation owned by Rust;
//!   the C side must call `huub_model_free`. Handles are **not thread-
//!   safe** — Huub's `Solver` is `!Send`, so each handle stays on the
//!   thread it was created on. (See `tools/huub_eval/src/portfolio.rs`
//!   for the build-in-thread-closure workaround used by the harness.)
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
    model::{Model, View, expressions::IntLinearExp},
    solver::{Solver, Status as HuubStatus, TerminationSignal, Valuation, View as SolverView},
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
    /// State machine: Building → Solved | NotSolved. Once solved we keep
    /// the assignment vectors so callers can read values; the underlying
    /// `Solver` is dropped because it's `!Send` and we don't want it to
    /// outlive the solve.
    inner: HandleState,
    /// Model-side int var registry. Index = C-visible var id.
    int_vars: Vec<View<IntVal>>,
    /// Model-side bool var registry.
    bool_vars: Vec<View<bool>>,
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
    Solved {
        result: HuubResult,
        int_vals: Vec<Option<IntVal>>,
        bool_vals: Vec<Option<bool>>,
    },
}

impl HuubModel {
    fn model_mut(&mut self) -> Result<&mut Model, &'static str> {
        match &mut self.inner {
            HandleState::Building(m) => Ok(m),
            HandleState::Solved { .. } => {
                Err("model already solved; further mutations are not allowed")
            }
        }
    }
}

// ----- Lifecycle ---------------------------------------------------------

/// Create a new model handle. Returns `NULL` on failure (call
/// `huub_last_error()`).
#[unsafe(no_mangle)]
pub extern "C" fn huub_model_new() -> *mut HuubModel {
    clear_error();
    let r = catch_unwind(|| {
        Box::into_raw(Box::new(HuubModel {
            inner: HandleState::Building(Box::new(Model::default())),
            int_vars: Vec::new(),
            bool_vars: Vec::new(),
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

/// Free a model handle previously returned by `huub_model_new()`. Passing
/// `NULL` is a no-op.
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

/// Post a linear constraint: `sum(coeffs[i] * int_var[var_ids[i]]) op rhs`.
///
/// `var_ids` and `coeffs` are arrays of length `n`. All ids must refer to
/// integer variables previously returned by `huub_model_new_int_var()`.
///
/// Returns `HuubResult::Satisfied` on success (the constraint was posted
/// without immediate conflict), `HuubResult::Unsatisfiable` if posting
/// the constraint immediately proved the model infeasible, or
/// `HuubResult::Error` on FFI failure.
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

// ----- Solve -------------------------------------------------------------

/// Lower the model and run a single-worker satisfaction search.
///
/// `time_limit_seconds <= 0` disables the wall-clock limit. Returns one of
/// `Satisfied`, `Unsatisfiable`, `Unknown`, or `Error`. After a successful
/// solve, call `huub_model_value_int` / `huub_model_value_bool` to read
/// the assignment.
///
/// Once called, the model transitions to a "solved" state and further
/// mutations (var creation, constraint posting) will fail.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_solve(
    handle: *mut HuubModel,
    time_limit_seconds: f64,
) -> HuubResult {
    clear_error();
    let res = with_handle(handle, |h| {
        // Short-circuit if a prior post-constraint call latched an
        // inconsistency. Avoids handing the SAT solver an under-
        // constrained model that would otherwise report SAT with an
        // assignment violating the not-actually-installed constraint.
        if h.known_unsat {
            h.inner = HandleState::Solved {
                result: HuubResult::Unsatisfiable,
                int_vals: vec![None; h.int_vars.len()],
                bool_vals: vec![None; h.bool_vars.len()],
            };
            return Ok(HuubResult::Unsatisfiable);
        }
        // Take the model out — lowering consumes it (we need owned access).
        let model = match std::mem::replace(
            &mut h.inner,
            HandleState::Solved {
                result: HuubResult::Error,
                int_vals: Vec::new(),
                bool_vals: Vec::new(),
            },
        ) {
            HandleState::Building(m) => *m,
            HandleState::Solved { result, .. } => {
                // Restore the solved state so subsequent value reads still work.
                h.inner = HandleState::Solved {
                    result,
                    int_vals: Vec::new(),
                    bool_vals: Vec::new(),
                };
                return Err("huub_model_solve: model already solved".to_string());
            }
        };

        let int_views = std::mem::take(&mut h.int_vars);
        let bool_views = std::mem::take(&mut h.bool_vars);

        let (final_result, int_vals, bool_vals) = solve_inner(model, &int_views, &bool_views, time_limit_seconds);
        h.inner = HandleState::Solved {
            result: final_result,
            int_vals,
            bool_vals,
        };
        Ok(final_result)
    });
    res.unwrap_or(HuubResult::Error)
}

fn solve_inner(
    mut model: Model,
    int_views: &[View<IntVal>],
    bool_views: &[View<bool>],
    time_limit_seconds: f64,
) -> (HuubResult, Vec<Option<IntVal>>, Vec<Option<bool>>) {
    // Lower. Mirrors `tools/huub_eval/src/translate.rs::solve_with_cfg`:
    // a conflict during lowering means UNSAT, not an FFI error.
    let lower_result = model.lower().to_solver();
    let (mut solver, map): (Solver, _) = match lower_result {
        Ok(pair) => pair,
        Err(e) => {
            let msg = format!("{e:?}");
            if msg.contains("Simplification(Conflict") || msg.contains("Conflict {") {
                return (HuubResult::Unsatisfiable, Vec::new(), Vec::new());
            }
            set_error(format!("huub_model_solve: lowering failed: {msg}"));
            return (HuubResult::Error, Vec::new(), Vec::new());
        }
    };

    // Wall-clock terminate callback. Same pattern as huub_eval.
    if time_limit_seconds > 0.0 {
        let deadline = Instant::now() + Duration::from_secs_f64(time_limit_seconds);
        solver.set_terminate_callback(Some(move || {
            if Instant::now() >= deadline {
                TerminationSignal::Terminate
            } else {
                TerminationSignal::Continue
            }
        }));
    }

    // Resolve model-side views to solver-side views. `LoweringMap::get`
    // returns `solver::View<T>`, which is the type that implements the
    // `Valuation` trait we need for reading values out of a `Solution`.
    let int_solver_views: Vec<SolverView<IntVal>> =
        int_views.iter().map(|v| map.get(&mut solver, *v)).collect();
    let bool_solver_views: Vec<SolverView<bool>> = bool_views
        .iter()
        .map(|v| map.get(&mut solver, *v))
        .collect();

    let mut int_vals: Vec<Option<IntVal>> = vec![None; int_solver_views.len()];
    let mut bool_vals: Vec<Option<bool>> = vec![None; bool_solver_views.len()];

    let status = solver
        .solve()
        .on_solution(|sol| {
            for (i, v) in int_solver_views.iter().enumerate() {
                int_vals[i] = Some(Valuation::val(v, sol));
            }
            for (i, v) in bool_solver_views.iter().enumerate() {
                bool_vals[i] = Some(Valuation::val(v, sol));
            }
        })
        .satisfy();

    let r = match status {
        HuubStatus::Satisfied => HuubResult::Satisfied,
        HuubStatus::Unsatisfiable => HuubResult::Unsatisfiable,
        HuubStatus::Complete => HuubResult::Complete,
        HuubStatus::Unknown => HuubResult::Unknown,
    };
    (r, int_vals, bool_vals)
}

// ----- Value extraction --------------------------------------------------

/// Read the value of an integer variable from the most recent solve.
/// Writes to `*out` and returns `Satisfied` on success; returns
/// `NotSolved` / `Error` on failure (leaving `*out` untouched).
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_value_int(
    handle: *mut HuubModel,
    var_id: i32,
    out: *mut i64,
) -> HuubResult {
    clear_error();
    let r = with_handle(handle, |h| match &h.inner {
        HandleState::Building(_) => Ok(HuubResult::NotSolved),
        HandleState::Solved { int_vals, .. } => {
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
#[unsafe(no_mangle)]
pub unsafe extern "C" fn huub_model_value_bool(
    handle: *mut HuubModel,
    var_id: i32,
    out: *mut bool,
) -> HuubResult {
    clear_error();
    let r = with_handle(handle, |h| match &h.inner {
        HandleState::Building(_) => Ok(HuubResult::NotSolved),
        HandleState::Solved { bool_vals, .. } => {
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
}
