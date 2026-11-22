//! Gradients of a function that doesn't exist until runtime: a Rust expression tree, built from
//! data, compiled to MLIR with melior and differentiated by Enzyme.
//!
//! `cargo run --example expression`
//!
//! This is what a frontend does: its programs come from its input, so there's no source to
//! differentiate ahead of time. Here the expression is f(x) = Σᵢ cᵢ sin(xᵢ xᵢ₊₁) + x₀², whose
//! length follows the coefficients `c`, and its gradient is checked against one computed by
//! walking the tree in Rust.

mod common;

use melior::{
    Context,
    dialect::{arith, func, memref},
    ir::{
        Block, BlockLike, Location, Module, OperationRef, Region, RegionLike, Type, Value,
        attribute::{FloatAttribute, IntegerAttribute, StringAttribute, TypeAttribute},
        operation::{OperationBuilder, OperationLike},
        r#type::{FunctionType, MemRefType},
    },
    pass::PassManager,
};
use melior_autodiff::{Activity, autodiff, pass, utility::create_context};

/// An expression in the variables x₀, x₁, ...
enum Expr {
    Var(usize),
    Const(f64),
    Add(Box<Expr>, Box<Expr>),
    Mul(Box<Expr>, Box<Expr>),
    Sin(Box<Expr>),
}

use Expr::*;

fn add(a: Expr, b: Expr) -> Expr {
    Add(Box::new(a), Box::new(b))
}

fn mul(a: Expr, b: Expr) -> Expr {
    Mul(Box::new(a), Box::new(b))
}

/// Σᵢ cᵢ sin(xᵢ xᵢ₊₁) + x₀², with one term per coefficient.
fn build(coefficients: &[f64]) -> Expr {
    let square = mul(Var(0), Var(0));
    coefficients
        .iter()
        .enumerate()
        .fold(square, |sum, (i, &c)| {
            let term = mul(Const(c), Sin(Box::new(mul(Var(i), Var(i + 1)))));
            add(sum, term)
        })
}

/// The expression's value and gradient at `x`, by forward-mode differentiation of the tree.
fn evaluate(expr: &Expr, x: &[f64]) -> (f64, Vec<f64>) {
    let n = x.len();
    match expr {
        Var(i) => {
            let mut gradient = vec![0.0; n];
            gradient[*i] = 1.0;
            (x[*i], gradient)
        }
        Const(c) => (*c, vec![0.0; n]),
        Add(a, b) => {
            let ((va, ga), (vb, gb)) = (evaluate(a, x), evaluate(b, x));
            (va + vb, ga.iter().zip(&gb).map(|(p, q)| p + q).collect())
        }
        Mul(a, b) => {
            let ((va, ga), (vb, gb)) = (evaluate(a, x), evaluate(b, x));
            let gradient = ga.iter().zip(&gb).map(|(p, q)| p * vb + va * q).collect();
            (va * vb, gradient)
        }
        Sin(a) => {
            let (va, ga) = evaluate(a, x);
            (va.sin(), ga.iter().map(|p| p * va.cos()).collect())
        }
    }
}

/// Appends the ops computing `expr` to `block`, whose arguments are the variables, and returns
/// the result.
fn emit<'c, 'a>(context: &'c Context, block: &'a Block<'c>, expr: &Expr) -> Value<'c, 'a> {
    let location = Location::unknown(context);
    let f64_type = Type::float64(context);
    let operation = match expr {
        Var(i) => return block.argument(*i).unwrap().into(),
        Const(c) => arith::constant(
            context,
            FloatAttribute::new(context, f64_type, *c).into(),
            location,
        ),
        Add(a, b) => arith::addf(emit(context, block, a), emit(context, block, b), location),
        Mul(a, b) => arith::mulf(emit(context, block, a), emit(context, block, b), location),
        Sin(a) => OperationBuilder::new("math.sin", location)
            .add_operands(&[emit(context, block, a)])
            .add_results(&[f64_type])
            .build()
            .unwrap(),
    };
    result(block.append_operation(operation))
}

fn result<'c, 'a>(operation: OperationRef<'c, 'a>) -> Value<'c, 'a> {
    operation.result(0).unwrap().into()
}

fn main() {
    let context = create_context();
    let location = Location::unknown(&context);
    let f64_type = Type::float64(&context);

    // The program's input: one term per coefficient, over coefficients.len() + 1 variables.
    let coefficients = [3.0, -1.5, 0.5];
    let expr = build(&coefficients);
    let n = coefficients.len() + 1;

    // func.func @f(%x0: f64, ..., %x3: f64) -> f64, its body emitted from the tree.
    let mut module = Module::new(location);
    let inputs = vec![f64_type; n];
    let block = Block::new(&inputs, &vec![location; n]).unwrap();
    let value = emit(&context, &block, &expr);
    block.append_operation(func::r#return(&[value], location));
    let region = Region::new();
    region.append_block(block);
    module.body().append_operation(func::func(
        &context,
        StringAttribute::new(&context, "f"),
        TypeAttribute::new(FunctionType::new(&context, &inputs, &[f64_type]).into()),
        region,
        &[],
        location,
    ));

    // func.func @grad_f(%x0: f64, ..., %x3: f64, %out: memref<4xf64>), storing
    // enzyme.autodiff @f's n results, the gradient, into %out.
    let array = MemRefType::new(f64_type, &[n as i64], None, None).into();
    let mut grad_inputs = inputs.clone();
    grad_inputs.push(array);
    common::add_function(&context, &module, "grad_f", &grad_inputs, |block| {
        let mut operands: Vec<Value> = (0..n).map(|i| block.argument(i).unwrap().into()).collect();
        // The incoming gradient of the result: 1.0, for the plain gradient.
        let one = arith::constant(
            &context,
            FloatAttribute::new(&context, f64_type, 1.0).into(),
            location,
        );
        operands.push(result(block.append_operation(one)));
        let gradient = block.append_operation(autodiff(
            &context,
            "f",
            &operands,
            &inputs,
            &vec![Activity::Active; n],
            &[Activity::ActiveNoNeed],
            1,
            false,
            location,
        ));
        let out = block.argument(n).unwrap().into();
        let index_type = Type::index(&context);
        for i in 0..n {
            let index = arith::constant(
                &context,
                IntegerAttribute::new(index_type, i as i64).into(),
                location,
            );
            let index = result(block.append_operation(index));
            let value = gradient.result(i).unwrap().into();
            block.append_operation(memref::store(value, out, &[index], location));
        }
    });

    assert!(module.as_operation().verify(), "invalid MLIR");
    let pm = PassManager::new(&context);
    pass::add_enzyme_lowering(&pm);
    pm.run(&mut module).expect("differentiation failed");
    let engine = common::jit(&context, &mut module);

    // grad_f(x0, x1, x2, x3, out): `out` is overwritten with ∇f. Calling it from Rust needs its
    // exact signature, so this is the one place the 3 coefficients above are hard-coded.
    let grad_f: extern "C" fn(f64, f64, f64, f64, *mut f64) =
        unsafe { common::lookup(&engine, "grad_f") };
    let x = [0.5, -1.0, 2.0, 0.25];
    let mut gradient = [0.0; 4];
    grad_f(x[0], x[1], x[2], x[3], gradient.as_mut_ptr());

    let (_, expected) = evaluate(&expr, &x);
    println!("∇f({x:?}) = {gradient:.6?}");
    for (got, want) in gradient.iter().zip(&expected) {
        assert!(
            (got - want).abs() < 1e-12,
            "got {gradient:?}, expected {expected:?}"
        );
    }
}
