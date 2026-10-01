use melior::pass::conversion::{
    create_lower_affine, create_reconcile_unrealized_casts, create_scf_to_control_flow,
    create_to_llvm,
};
use melior::pass::linalg::create_convert_linalg_to_loops_pass;
use melior::pass::transform::{
    create_canonicalizer_pass, create_cse_pass, create_inliner_pass, create_symbol_dce_pass,
};
use melior::pass::PassManager;
use melior::{ir::Module, Context, ExecutionEngine};
use melior_autodiff::{
    enzymeCreateConvertEnzymeToMemRefPass, enzymeCreateDifferentiatePass, Activity,
};

mod common;
use common::{gen_autodiff, gen_fwddiff, setup_context};

// ── Pipeline ─────────────────────────────────────────────────────────────────

fn compile<'a>(ctx: &'a Context, src: &str) -> (Module<'a>, ExecutionEngine) {
    let mut module = Module::parse(ctx, src).expect("parse failed");
    derive_linalg_wrappers(ctx, &module, src);
    let pm = PassManager::new(ctx);
    pm.add_pass(unsafe { melior::pass::Pass::from_raw_fn(enzymeCreateDifferentiatePass) });
    pm.add_pass(unsafe { melior::pass::Pass::from_raw_fn(enzymeCreateConvertEnzymeToMemRefPass) });
    pm.add_pass(create_inliner_pass());
    pm.add_pass(create_canonicalizer_pass());
    pm.add_pass(create_cse_pass());
    pm.add_pass(create_symbol_dce_pass());
    pm.add_pass(create_scf_to_control_flow());
    pm.add_pass(create_to_llvm());
    pm.add_pass(create_reconcile_unrealized_casts());
    pm.run(&mut module).expect("lowering failed");
    let engine = ExecutionEngine::new(&module, 2, &[], false, false);
    (module, engine)
}

// Pipeline that lowers linalg ops to loops first, then runs Enzyme.
fn compile_linalg<'a>(ctx: &'a Context, src: &str) -> (Module<'a>, ExecutionEngine) {
    let mut module = Module::parse(ctx, src).expect("parse failed");
    derive_linalg_wrappers(ctx, &module, src);
    let pm = PassManager::new(ctx);
    pm.add_pass(create_convert_linalg_to_loops_pass());
    pm.add_pass(create_lower_affine());
    pm.add_pass(create_canonicalizer_pass());
    pm.add_pass(unsafe { melior::pass::Pass::from_raw_fn(enzymeCreateDifferentiatePass) });
    pm.add_pass(unsafe {
        melior::pass::Pass::from_raw_fn(enzymeCreateConvertEnzymeToMemRefPass)
    });
    pm.add_pass(create_inliner_pass());
    pm.add_pass(create_canonicalizer_pass());
    pm.add_pass(create_cse_pass());
    pm.add_pass(create_symbol_dce_pass());
    pm.add_pass(create_scf_to_control_flow());
    pm.add_pass(create_to_llvm());
    pm.add_pass(create_reconcile_unrealized_casts());
    pm.run(&mut module).expect("lowering failed");
    let engine = ExecutionEngine::new(&module, 2, &[], false, false);
    (module, engine)
}

fn derive_linalg_wrappers(ctx: &Context, module: &Module<'_>, src: &str) {
    if src.contains("func.func @dot4") {
        gen_fwddiff(
            ctx,
            module,
            "jvp_dot4",
            "dot4",
            &[
                "memref<4xf64>",
                "memref<4xf64>",
                "memref<4xf64>",
                "memref<4xf64>",
            ],
            &["f64"],
            &[Activity::Dup, Activity::Dup],
            &[Activity::DupNoNeed],
            true,
        );
    }
    if src.contains("func.func @cube") {
        gen_autodiff(
            ctx,
            module,
            "dcube",
            "cube",
            &["f64", "f64"],
            &["f64"],
            &[Activity::Active],
            &[Activity::ActiveNoNeed],
            true,
        );
    }
    if src.contains("func.func @poly") {
        gen_autodiff(
            ctx,
            module,
            "dpoly",
            "poly",
            &["f64", "f64"],
            &["f64"],
            &[Activity::Active],
            &[Activity::ActiveNoNeed],
            true,
        );
    }
    if src.contains("func.func @prod") {
        gen_autodiff(
            ctx,
            module,
            "dprod_dx",
            "prod",
            &["f64", "f64", "f64", "f64"],
            &["f64"],
            &[Activity::Active, Activity::Const],
            &[Activity::ActiveNoNeed],
            true,
        );
    }
    if src.contains("func.func @trace4") {
        gen_fwddiff(
            ctx,
            module,
            "jvp_trace4",
            "trace4",
            &["memref<4x4xf64>", "memref<4x4xf64>"],
            &["f64"],
            &[Activity::Dup],
            &[Activity::DupNoNeed],
            true,
        );
    }
    if src.contains("func.func @matmul_sum") {
        // JVP wrt both A and B: jvp_matmul_sum(A, dA, B, dB) -> f64
        gen_fwddiff(
            ctx,
            module,
            "jvp_matmul_sum",
            "matmul_sum",
            &[
                "memref<2x2xf64>",
                "memref<2x2xf64>",
                "memref<2x2xf64>",
                "memref<2x2xf64>",
            ],
            &["f64"],
            &[Activity::Dup, Activity::Dup],
            &[Activity::DupNoNeed],
            true,
        );
    }
    if src.contains("func.func @sum_sq") {
        // JVP wrt x: jvp_sum_sq(x, dx) -> f64
        gen_fwddiff(
            ctx,
            module,
            "jvp_sum_sq",
            "sum_sq",
            &["memref<4xf64>", "memref<4xf64>"],
            &["f64"],
            &[Activity::Dup],
            &[Activity::DupNoNeed],
            true,
        );
    }
}

// Call a JIT-compiled (f64, f64) -> f64 function by name.
fn call2(engine: &ExecutionEngine, name: &str, x: f64, y: f64) -> f64 {
    let mut x = x;
    let mut y = y;
    let mut out = 0.0_f64;
    unsafe {
        engine
            .invoke_packed(
                name,
                &mut [
                    &mut x as *mut f64 as *mut (),
                    &mut y as *mut f64 as *mut (),
                    &mut out as *mut f64 as *mut (),
                ],
            )
            .unwrap_or_else(|_| panic!("{name} invoke failed"));
    }
    out
}

// Central finite difference: (f(x+h) - f(x-h)) / 2h
fn finite_diff(f: impl Fn(f64) -> f64, x: f64) -> f64 {
    let h = 1e-5;
    (f(x + h) - f(x - h)) / (2.0 * h)
}

fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-6
}

// ── Array helpers ─────────────────────────────────────────────────────────────

// Generic MemRef descriptor matching MLIR's LLVM lowering for ranked memrefs:
// allocated ptr, aligned ptr, offset, then sizes[R], then strides[R].
#[repr(C)]
struct MemRef<const R: usize> {
    allocated: *mut f64,
    aligned: *mut f64,
    offset: i64,
    sizes: [i64; R],
    strides: [i64; R],
}

impl MemRef<1> {
    fn from_slice(data: &mut [f64]) -> Self {
        MemRef {
            allocated: data.as_mut_ptr(),
            aligned: data.as_mut_ptr(),
            offset: 0,
            sizes: [data.len() as i64],
            strides: [1],
        }
    }
}

impl MemRef<2> {
    fn from_2d<const M: usize, const N: usize>(data: &mut [[f64; N]; M]) -> Self {
        MemRef {
            allocated: data.as_mut_ptr() as *mut f64,
            aligned: data.as_mut_ptr() as *mut f64,
            offset: 0,
            sizes: [M as i64, N as i64],
            strides: [N as i64, 1],
        }
    }
}

type DotFn = unsafe extern "C" fn(*mut MemRef<1>, *mut MemRef<1>) -> f64;
type JvpDotFn =
    unsafe extern "C" fn(*mut MemRef<1>, *mut MemRef<1>, *mut MemRef<1>, *mut MemRef<1>) -> f64;

fn lookup_dot(engine: &ExecutionEngine, name: &str) -> DotFn {
    let sym = format!("_mlir_ciface_{name}");
    unsafe { std::mem::transmute(engine.lookup(&sym)) }
}

fn lookup_jvp(engine: &ExecutionEngine, name: &str) -> JvpDotFn {
    let sym = format!("_mlir_ciface_{name}");
    unsafe { std::mem::transmute(engine.lookup(&sym)) }
}

// Forward-mode directional derivative via a direct fn-ptr call.
fn jvp_dot<const N: usize>(
    f: JvpDotFn,
    x: &mut [f64; N],
    y: &mut [f64; N],
    direction_x: &mut [f64; N],
) -> f64 {
    let mut dy = [0.0f64; N];
    let mut xd = MemRef::<1>::from_slice(x);
    let mut dxd = MemRef::<1>::from_slice(direction_x);
    let mut yd = MemRef::<1>::from_slice(y);
    let mut dyd = MemRef::<1>::from_slice(&mut dy);
    unsafe { f(&mut xd, &mut dxd, &mut yd, &mut dyd) }
}

// Compute the full gradient of f(x, y_fixed) via N JVP calls (one per basis vector).
fn grad_via_jvp<const N: usize>(f: JvpDotFn, x: &mut [f64; N], y: &mut [f64; N]) -> [f64; N] {
    std::array::from_fn(|i| {
        let mut ei = [0.0f64; N];
        ei[i] = 1.0;
        jvp_dot(f, x, y, &mut ei)
    })
}

// Finite-difference gradient of dot(x, y) w.r.t. x.
fn fd_grad_1d<const N: usize>(f: DotFn, x: &[f64; N], y: &mut [f64; N]) -> [f64; N] {
    let h = 1e-5;
    std::array::from_fn(|i| {
        let mut xp = *x;
        let mut xm = *x;
        xp[i] += h;
        xm[i] -= h;
        let fp = unsafe {
            f(
                &mut MemRef::<1>::from_slice(&mut xp),
                &mut MemRef::<1>::from_slice(y),
            )
        };
        let fm = unsafe {
            f(
                &mut MemRef::<1>::from_slice(&mut xm),
                &mut MemRef::<1>::from_slice(y),
            )
        };
        (fp - fm) / (2.0 * h)
    })
}

// ── Tests ─────────────────────────────────────────────────────────────────────

// ── Array tests ───────────────────────────────────────────────────────────────

const DOT4_MODULE: &str = r#"
module {
  func.func @dot4(%x: memref<4xf64>, %y: memref<4xf64>) -> f64
      attributes { llvm.emit_c_interface } {
    %c0  = arith.constant 0 : index
    %c1  = arith.constant 1 : index
    %c4  = arith.constant 4 : index
    %z   = arith.constant 0.0 : f64
    %r   = scf.for %i = %c0 to %c4 step %c1 iter_args(%s = %z) -> f64 {
      %xi  = memref.load %x[%i] : memref<4xf64>
      %yi  = memref.load %y[%i] : memref<4xf64>
      %p   = arith.mulf %xi, %yi : f64
      %s2  = arith.addf %s, %p : f64
      scf.yield %s2 : f64
    }
    return %r : f64
  }
}
"#;

#[test]
fn print_dot4_ir() {
    let ctx = setup_context();
    let mut module = melior::ir::Module::parse(
        &ctx,
        r#"
module {
  func.func @dot4(%x: memref<4xf64>, %y: memref<4xf64>) -> f64
      attributes { llvm.emit_c_interface } {
    %c0  = arith.constant 0 : index
    %c1  = arith.constant 1 : index
    %c4  = arith.constant 4 : index
    %z   = arith.constant 0.0 : f64
    %r   = scf.for %i = %c0 to %c4 step %c1 iter_args(%s = %z) -> f64 {
      %xi  = memref.load %x[%i] : memref<4xf64>
      %yi  = memref.load %y[%i] : memref<4xf64>
      %p   = arith.mulf %xi, %yi : f64
      %s2  = arith.addf %s, %p : f64
      scf.yield %s2 : f64
    }
    return %r : f64
  }
}
"#,
    )
    .expect("parse failed");
    let pm = melior::pass::PassManager::new(&ctx);
    pm.add_pass(melior::pass::transform::create_canonicalizer_pass());
    pm.add_pass(melior::pass::conversion::create_scf_to_control_flow());
    pm.add_pass(melior::pass::conversion::create_to_llvm());
    pm.add_pass(melior::pass::conversion::create_reconcile_unrealized_casts());
    pm.run(&mut module).expect("lowering failed");
    eprintln!("{}", module.as_operation());
}

// dot4([1,2,3,4], [5,6,7,8]) = 1*5 + 2*6 + 3*7 + 4*8 = 70
#[test]
fn primal_dot4() {
    let ctx = setup_context();
    let (_, engine) = compile(&ctx, DOT4_MODULE);
    let mut x = [1.0, 2.0, 3.0, 4.0f64];
    let mut y = [5.0, 6.0, 7.0, 8.0f64];
    let dot = lookup_dot(&engine, "dot4");
    let out = unsafe {
        dot(
            &mut MemRef::<1>::from_slice(&mut x),
            &mut MemRef::<1>::from_slice(&mut y),
        )
    };
    assert!(approx(out, 70.0), "dot4 = {out}, expected 70");
}

// ∂dot(x,y)/∂x = y,  verified via forward-mode JVP against finite differences.
#[test]
fn grad_dot_wrt_x() {
    let ctx = setup_context();
    let (_, engine) = compile(&ctx, DOT4_MODULE);

    let mut x = [1.0, 2.0, 3.0, 4.0f64];
    let mut y = [5.0, 6.0, 7.0, 8.0f64];

    let jvp = lookup_jvp(&engine, "jvp_dot4");
    let dot = lookup_dot(&engine, "dot4");
    let enzyme = grad_via_jvp(jvp, &mut x, &mut y);
    let fd = fd_grad_1d(dot, &x, &mut y);

    for i in 0..4 {
        assert!(
            approx(enzyme[i], fd[i]),
            "i={i}: enzyme={}, fd={}",
            enzyme[i],
            fd[i]
        );
        assert!(
            approx(enzyme[i], y[i]),
            "∂dot/∂x[{i}] should equal y[{i}]={}",
            y[i]
        );
    }
}

// ∂dot(x,y)/∂y = x,  swap roles.
#[test]
fn grad_dot_wrt_y() {
    let ctx = setup_context();
    let (_, engine) = compile(&ctx, DOT4_MODULE);

    let mut x = [1.0, 2.0, 3.0, 4.0f64];
    let mut y = [5.0, 6.0, 7.0, 8.0f64];

    let jvp = lookup_jvp(&engine, "jvp_dot4");
    let dot = lookup_dot(&engine, "dot4");
    // Swap x and y in the JVP call to get ∂/∂y.
    let enzyme = grad_via_jvp(jvp, &mut y, &mut x);
    let fd = fd_grad_1d(dot, &y, &mut x);

    for i in 0..4 {
        assert!(
            approx(enzyme[i], fd[i]),
            "i={i}: enzyme={}, fd={}",
            enzyme[i],
            fd[i]
        );
        assert!(
            approx(enzyme[i], x[i]),
            "∂dot/∂y[{i}] should equal x[{i}]={}",
            x[i]
        );
    }
}

// f(x) = x³,  f'(x) = 3x²
#[test]
fn grad_cube() {
    let ctx = setup_context();
    let (_, engine) = compile(
        &ctx,
        r#"
module {
  func.func @cube(%x: f64) -> f64 {
    %x2 = arith.mulf %x, %x : f64
    %x3 = arith.mulf %x2, %x : f64
    return %x3 : f64
  }
}
"#,
    );

    for x in [-2.0, -1.0, 0.5, 1.0, 3.0] {
        let enzyme = call2(&engine, "dcube", x, 1.0);
        let fd = finite_diff(|t| t * t * t, x);
        assert!(approx(enzyme, fd), "x={x}: enzyme={enzyme}, fd={fd}");
    }
}

// f(x) = x² - 3x + 1,  f'(x) = 2x - 3
#[test]
fn grad_polynomial() {
    let ctx = setup_context();
    let (_, engine) = compile(
        &ctx,
        r#"
module {
  func.func @poly(%x: f64) -> f64 {
    %c3 = arith.constant 3.0 : f64
    %c1 = arith.constant 1.0 : f64
    %x2 = arith.mulf %x, %x : f64
    %t  = arith.mulf %c3, %x : f64
    %y  = arith.subf %x2, %t : f64
    %y1 = arith.addf %y, %c1 : f64
    return %y1 : f64
  }
}
"#,
    );

    for x in [-2.0, 0.0, 1.5, 3.0] {
        let enzyme = call2(&engine, "dpoly", x, 1.0);
        let fd = finite_diff(|t| t * t - 3.0 * t + 1.0, x);
        assert!(approx(enzyme, fd), "x={x}: enzyme={enzyme}, fd={fd}");
    }
}

// f(x, y) = x * y,  ∂f/∂x = y,  ∂f/∂y = x
#[test]
fn grad_product_wrt_x() {
    let ctx = setup_context();
    let (_, engine) = compile(
        &ctx,
        r#"
module {
  func.func @prod(%x: f64, %y: f64) -> f64 {
    %r = arith.mulf %x, %y : f64
    return %r : f64
  }
}
"#,
    );

    for (x, y) in [(1.0, 2.0), (-3.0, 4.0), (0.5, 0.5)] {
        let mut xa = x;
        let mut dxa = 1.0_f64;
        let mut ya = y;
        let mut dya = 0.0_f64;
        let mut out = 0.0_f64;
        unsafe {
            engine
                .invoke_packed(
                    "dprod_dx",
                    &mut [
                        &mut xa as *mut f64 as *mut (),
                        &mut dxa as *mut f64 as *mut (),
                        &mut ya as *mut f64 as *mut (),
                        &mut dya as *mut f64 as *mut (),
                        &mut out as *mut f64 as *mut (),
                    ],
                )
                .expect("dprod_dx invoke failed");
        }
        assert!(
            approx(out, y),
            "∂(x*y)/∂x at ({x},{y}): enzyme={out}, expected={y}"
        );
    }
}

// ── 2D array tests ────────────────────────────────────────────────────────────

const TRACE4_MODULE: &str = r#"
module {
  func.func @trace4(%A: memref<4x4xf64>) -> f64
      attributes { llvm.emit_c_interface } {
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c4 = arith.constant 4 : index
    %z  = arith.constant 0.0 : f64
    %r  = scf.for %i = %c0 to %c4 step %c1 iter_args(%s = %z) -> f64 {
      %a  = memref.load %A[%i, %i] : memref<4x4xf64>
      %s2 = arith.addf %s, %a : f64
      scf.yield %s2 : f64
    }
    return %r : f64
  }
}
"#;

type TraceFn = unsafe extern "C" fn(*mut MemRef<2>) -> f64;
type JvpTraceFn = unsafe extern "C" fn(*mut MemRef<2>, *mut MemRef<2>) -> f64;

// trace(I₄) = 4
#[test]
fn primal_trace4() {
    let ctx = setup_context();
    let (_, engine) = compile(&ctx, TRACE4_MODULE);
    let f: TraceFn = unsafe { std::mem::transmute(engine.lookup("_mlir_ciface_trace4")) };
    let mut eye: [[f64; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let out = unsafe { f(&mut MemRef::<2>::from_2d(&mut eye)) };
    assert!(approx(out, 4.0), "trace(I) = {out}, expected 4");
}

// trace is linear: fwddiff(@trace4, A, dA) = trace(dA).
// With dA = I: result = 4.  With dA = e_{ij} (i≠j): result = 0.
#[test]
fn grad_trace4() {
    let ctx = setup_context();
    let (_, engine) = compile(&ctx, TRACE4_MODULE);
    let jvp: JvpTraceFn = unsafe { std::mem::transmute(engine.lookup("_mlir_ciface_jvp_trace4")) };

    let mut a: [[f64; 4]; 4] = [
        [2.0, 1.0, 0.0, 0.0],
        [0.0, 3.0, 1.0, 0.0],
        [0.0, 0.0, 4.0, 1.0],
        [0.0, 0.0, 0.0, 5.0],
    ];

    // dA = I  →  trace(dA) = 4
    let mut eye: [[f64; 4]; 4] = [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ];
    let r = unsafe {
        jvp(
            &mut MemRef::<2>::from_2d(&mut a),
            &mut MemRef::<2>::from_2d(&mut eye),
        )
    };
    assert!(approx(r, 4.0), "jvp with dA=I: {r}, expected 4");

    // dA = e_{01}  →  trace(dA) = 0  (off-diagonal)
    let mut e01: [[f64; 4]; 4] = [[0.0; 4]; 4];
    e01[0][1] = 1.0;
    let r = unsafe {
        jvp(
            &mut MemRef::<2>::from_2d(&mut a),
            &mut MemRef::<2>::from_2d(&mut e01),
        )
    };
    assert!(approx(r, 0.0), "jvp with dA=e_01: {r}, expected 0");

    // dA = e_{22}  →  trace(dA) = 1  (diagonal)
    let mut e22: [[f64; 4]; 4] = [[0.0; 4]; 4];
    e22[2][2] = 1.0;
    let r = unsafe {
        jvp(
            &mut MemRef::<2>::from_2d(&mut a),
            &mut MemRef::<2>::from_2d(&mut e22),
        )
    };
    assert!(approx(r, 1.0), "jvp with dA=e_22: {r}, expected 1");
}

// ── linalg.matmul tests ───────────────────────────────────────────────────────

// f(A, B) = sum_all(A @ B), where A, B ∈ ℝ^{2×2}.
// df/dA_{pq} = Σ_j B[q][j]  (row-sum of B's q-th row, broadcast over p)
// df/dB_{pq} = Σ_i A[i][p]  (col-sum of A's p-th col, broadcast over q)
const MATMUL_SUM_MODULE: &str = r#"
module {
  func.func @matmul_sum(%A: memref<2x2xf64>, %B: memref<2x2xf64>) -> f64
      attributes { llvm.emit_c_interface } {
    %C = memref.alloc() : memref<2x2xf64>
    %zero = arith.constant 0.0 : f64
    linalg.fill ins(%zero : f64) outs(%C : memref<2x2xf64>)
    linalg.matmul ins(%A, %B : memref<2x2xf64>, memref<2x2xf64>)
                  outs(%C : memref<2x2xf64>)
    %c0 = arith.constant 0 : index
    %c1 = arith.constant 1 : index
    %c2 = arith.constant 2 : index
    %s0 = scf.for %i = %c0 to %c2 step %c1 iter_args(%acc = %zero) -> f64 {
      %s1 = scf.for %j = %c0 to %c2 step %c1 iter_args(%acc2 = %acc) -> f64 {
        %v  = memref.load %C[%i, %j] : memref<2x2xf64>
        %s2 = arith.addf %acc2, %v : f64
        scf.yield %s2 : f64
      }
      scf.yield %s1 : f64
    }
    return %s0 : f64
  }
}
"#;

type JvpMatmulFn = unsafe extern "C" fn(
    *mut MemRef<2>,
    *mut MemRef<2>,
    *mut MemRef<2>,
    *mut MemRef<2>,
) -> f64;

// Primal: sum_all([[1,2],[3,4]] @ [[5,6],[7,8]])
// = sum_all([[1*5+2*7, 1*6+2*8], [3*5+4*7, 3*6+4*8]])
// = sum_all([[19, 22], [43, 50]]) = 134
#[test]
fn primal_matmul_sum() {
    let ctx = setup_context();
    let (_, engine) = compile_linalg(&ctx, MATMUL_SUM_MODULE);
    let f: unsafe extern "C" fn(*mut MemRef<2>, *mut MemRef<2>) -> f64 =
        unsafe { std::mem::transmute(engine.lookup("_mlir_ciface_matmul_sum")) };
    let mut a: [[f64; 2]; 2] = [[1.0, 2.0], [3.0, 4.0]];
    let mut b: [[f64; 2]; 2] = [[5.0, 6.0], [7.0, 8.0]];
    let out = unsafe {
        f(
            &mut MemRef::<2>::from_2d(&mut a),
            &mut MemRef::<2>::from_2d(&mut b),
        )
    };
    assert!(approx(out, 134.0), "matmul_sum = {out}, expected 134");
}

// d/dA sum_all(A @ B):  for B = [[1,2],[3,4]],
// (dF/dA)[p][0] = row_sum(B)[0] = 1+2 = 3
// (dF/dA)[p][1] = row_sum(B)[1] = 3+4 = 7
// so the full gradient matrix is [[3,7],[3,7]].
#[test]
fn grad_matmul_wrt_a() {
    let ctx = setup_context();
    let (_, engine) = compile_linalg(&ctx, MATMUL_SUM_MODULE);
    let jvp: JvpMatmulFn =
        unsafe { std::mem::transmute(engine.lookup("_mlir_ciface_jvp_matmul_sum")) };

    let mut a: [[f64; 2]; 2] = [[1.0, 2.0], [3.0, 4.0]];
    let mut b: [[f64; 2]; 2] = [[1.0, 2.0], [3.0, 4.0]];
    let mut db_zero: [[f64; 2]; 2] = [[0.0; 2]; 2];

    // Directional derivatives for each basis direction of A.
    // dA = e_{ij}  ⟹  JVP = (dF/dA)[i][j]
    let expected = [[3.0_f64, 7.0], [3.0, 7.0]];
    for i in 0..2 {
        for j in 0..2 {
            let mut da: [[f64; 2]; 2] = [[0.0; 2]; 2];
            da[i][j] = 1.0;
            let got = unsafe {
                jvp(
                    &mut MemRef::<2>::from_2d(&mut a),
                    &mut MemRef::<2>::from_2d(&mut da),
                    &mut MemRef::<2>::from_2d(&mut b),
                    &mut MemRef::<2>::from_2d(&mut db_zero),
                )
            };
            assert!(
                approx(got, expected[i][j]),
                "d/dA[{i}][{j}]: got {got}, expected {}",
                expected[i][j]
            );
        }
    }
}

// d/dB sum_all(A @ B):  for A = [[1,2],[3,4]],
// (dF/dB)[p][q] = col_sum(A)[p]
// col_sum(A): col 0 → 1+3=4, col 1 → 2+4=6
// so gradient is [[4,4],[6,6]].
#[test]
fn grad_matmul_wrt_b() {
    let ctx = setup_context();
    let (_, engine) = compile_linalg(&ctx, MATMUL_SUM_MODULE);
    let jvp: JvpMatmulFn =
        unsafe { std::mem::transmute(engine.lookup("_mlir_ciface_jvp_matmul_sum")) };

    let mut a: [[f64; 2]; 2] = [[1.0, 2.0], [3.0, 4.0]];
    let mut da_zero: [[f64; 2]; 2] = [[0.0; 2]; 2];
    let mut b: [[f64; 2]; 2] = [[1.0, 2.0], [3.0, 4.0]];

    let expected = [[4.0_f64, 4.0], [6.0, 6.0]];
    for i in 0..2 {
        for j in 0..2 {
            let mut db: [[f64; 2]; 2] = [[0.0; 2]; 2];
            db[i][j] = 1.0;
            let got = unsafe {
                jvp(
                    &mut MemRef::<2>::from_2d(&mut a),
                    &mut MemRef::<2>::from_2d(&mut da_zero),
                    &mut MemRef::<2>::from_2d(&mut b),
                    &mut MemRef::<2>::from_2d(&mut db),
                )
            };
            assert!(
                approx(got, expected[i][j]),
                "d/dB[{i}][{j}]: got {got}, expected {}",
                expected[i][j]
            );
        }
    }
}

// ── linalg.generic tests ──────────────────────────────────────────────────────

// f(x) = Σ x_i²  implemented with linalg.generic (reduction over 1-D memref).
// f'(x)_i = 2 x_i
const SUM_SQ_MODULE: &str = r#"
module {
  func.func @sum_sq(%x: memref<4xf64>) -> f64
      attributes { llvm.emit_c_interface } {
    %out = memref.alloc() : memref<f64>
    %zero = arith.constant 0.0 : f64
    memref.store %zero, %out[] : memref<f64>
    linalg.generic {
      indexing_maps = [affine_map<(d0) -> (d0)>, affine_map<(d0) -> ()>],
      iterator_types = ["reduction"]
    } ins(%x : memref<4xf64>) outs(%out : memref<f64>) {
    ^bb0(%xi: f64, %acc: f64):
      %sq  = arith.mulf %xi, %xi : f64
      %sum = arith.addf %acc, %sq : f64
      linalg.yield %sum : f64
    }
    %res = memref.load %out[] : memref<f64>
    return %res : f64
  }
}
"#;

// Primal: 1² + 2² + 3² + 4² = 30
#[test]
fn primal_sum_sq() {
    let ctx = setup_context();
    let (_, engine) = compile_linalg(&ctx, SUM_SQ_MODULE);
    let f: unsafe extern "C" fn(*mut MemRef<1>) -> f64 =
        unsafe { std::mem::transmute(engine.lookup("_mlir_ciface_sum_sq")) };
    let mut x = [1.0, 2.0, 3.0, 4.0f64];
    let out = unsafe { f(&mut MemRef::<1>::from_slice(&mut x)) };
    assert!(approx(out, 30.0), "sum_sq = {out}, expected 30");
}

// JVP of f(x) = Σ x_i² in direction e_i is 2 x_i.
// For x = [1,2,3,4]: gradient = [2,4,6,8].
#[test]
fn grad_sum_sq() {
    let ctx = setup_context();
    let (_, engine) = compile_linalg(&ctx, SUM_SQ_MODULE);
    let jvp: unsafe extern "C" fn(*mut MemRef<1>, *mut MemRef<1>) -> f64 =
        unsafe { std::mem::transmute(engine.lookup("_mlir_ciface_jvp_sum_sq")) };

    let mut x = [1.0, 2.0, 3.0, 4.0f64];
    let expected = [2.0, 4.0, 6.0, 8.0f64];
    for i in 0..4 {
        let mut ei = [0.0f64; 4];
        ei[i] = 1.0;
        let got = unsafe {
            jvp(
                &mut MemRef::<1>::from_slice(&mut x),
                &mut MemRef::<1>::from_slice(&mut ei),
            )
        };
        assert!(
            approx(got, expected[i]),
            "jvp_sum_sq direction e_{i}: got {got}, expected {}",
            expected[i]
        );
    }
}
