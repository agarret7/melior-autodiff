use melior_autodiff::{hmc_config_attr, nuts_config_attr, symbol_attr};
use mlir_sys::{mlirAttributePrint, MlirAttribute, MlirStringRef};

mod common;
use common::setup_context;

unsafe fn attr_to_string(attr: MlirAttribute) -> String {
    let mut s = String::new();
    unsafe extern "C" fn callback(sr: MlirStringRef, userdata: *mut std::ffi::c_void) {
        let s = unsafe { &mut *(userdata as *mut String) };
        let bytes = unsafe { std::slice::from_raw_parts(sr.data as *const u8, sr.length) };
        s.push_str(std::str::from_utf8(bytes).unwrap());
    }
    unsafe { mlirAttributePrint(attr, Some(callback), &mut s as *mut String as *mut std::ffi::c_void) };
    s
}

#[test]
fn hmc_config_attr_constructs() {
    let ctx = setup_context();
    let attr = unsafe { hmc_config_attr(ctx.to_raw(), 1.5, true, false) };
    assert!(!attr.ptr.is_null());
    let text = unsafe { attr_to_string(attr) };
    eprintln!("hmc_config_attr: {text}");
    assert!(text.contains("1.5") || text.contains("hmc"), "unexpected repr: {text}");
    assert!(!text.contains("nuts"), "HMC attr should not mention nuts: {text}");
}

#[test]
fn nuts_config_attr_with_max_delta_energy() {
    let ctx = setup_context();
    let attr = unsafe { nuts_config_attr(ctx.to_raw(), 10, Some(1000.0), true, true) };
    assert!(!attr.ptr.is_null());
    let text = unsafe { attr_to_string(attr) };
    eprintln!("nuts_config_attr (Some): {text}");
    assert!(!text.is_empty());
    assert!(!text.contains("hmc"), "NUTS attr should not mention hmc: {text}");
}

#[test]
fn nuts_config_attr_without_max_delta_energy() {
    let ctx = setup_context();
    let attr_with = unsafe { nuts_config_attr(ctx.to_raw(), 10, Some(1000.0), false, false) };
    let attr_without = unsafe { nuts_config_attr(ctx.to_raw(), 10, None, false, false) };
    assert!(!attr_with.ptr.is_null());
    assert!(!attr_without.ptr.is_null());
    let text_with = unsafe { attr_to_string(attr_with) };
    let text_without = unsafe { attr_to_string(attr_without) };
    eprintln!("nuts_config_attr (Some(1000.0)): {text_with}");
    eprintln!("nuts_config_attr (None):         {text_without}");
    assert_ne!(text_with, text_without, "Some and None max_delta_energy should produce different attrs");
}

#[test]
fn symbol_attr_constructs() {
    let ctx = setup_context();
    extern "C" fn dummy_logpdf() -> f64 { 0.0 }
    let ptr = dummy_logpdf as u64;
    let attr = unsafe { symbol_attr(ctx.to_raw(), ptr) };
    assert!(!attr.ptr.is_null());
    let text = unsafe { attr_to_string(attr) };
    eprintln!("symbol_attr: {text}");
    assert!(!text.is_empty());
}

#[test]
fn symbol_attr_encodes_pointer() {
    let ctx = setup_context();
    extern "C" fn fn_a() -> f64 { 1.0 }
    extern "C" fn fn_b() -> f64 { 2.0 }
    let attr_a = unsafe { symbol_attr(ctx.to_raw(), fn_a as u64) };
    let attr_b = unsafe { symbol_attr(ctx.to_raw(), fn_b as u64) };
    assert!(!attr_a.ptr.is_null());
    assert!(!attr_b.ptr.is_null());
    let text_a = unsafe { attr_to_string(attr_a) };
    let text_b = unsafe { attr_to_string(attr_b) };
    eprintln!("symbol_attr fn_a: {text_a}");
    eprintln!("symbol_attr fn_b: {text_b}");
    assert_ne!(text_a, text_b, "distinct function pointers should produce distinct symbol attrs");
}
