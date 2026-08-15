//! Does a cheap function of S decide the Fourier coefficient?
//!
//! `⟨Z_S⟩ = ⟨0| M'† Q_S M' |0⟩` with `Q_S = C† Z_S C` a single Pauli and
//! `M' = ∏ exp(-iπ/8 P_a)`. Each rotation leaves `Q` alone when it
//! commutes and splits it in two at equal weight `1/√2` when it does
//! not. So the coefficient should be governed by how many axes `Q_S`
//! anticommutes with — a count, computable per S by Clifford means.
use quantsim::dcs::{self, Dcs};
use quantsim::pathsum::Mask;
use quantsim::prelude::*;
use std::collections::BTreeMap;

fn fwht(v: &mut [f64]) {
    let n = v.len();
    let mut h = 1;
    while h < n {
        for i in (0..n).step_by(h * 2) {
            for j in i..i + h {
                let (a, b) = (v[j], v[j + h]);
                v[j] = a + b;
                v[j + h] = a - b;
            }
        }
        h *= 2;
    }
}

/// C† Z_q C for every q, from the skeleton's tableau.
fn z_images(circuit: &Circuit<C64>, n: usize) -> Vec<(Mask, Mask)> {
    let mut xim: Vec<(Mask, Mask)> = (0..n).map(|q| (Mask::single(q), Mask::zero())).collect();
    let mut zim: Vec<(Mask, Mask)> = (0..n).map(|q| (Mask::zero(), Mask::single(q))).collect();
    let mul = |a: &(Mask, Mask), b: &(Mask, Mask)| (a.0.xor(&b.0), a.1.xor(&b.1));
    for op in circuit.ops() {
        let Op::Named { name, qubits, .. } = op else { continue };
        match (name.as_str(), qubits.len()) {
            ("h", 1) => { let a = qubits[0]; xim.swap(a, a); std::mem::swap(&mut xim[a], &mut zim[a]); }
            ("s" | "sdg", 1) => { let a = qubits[0]; xim[a] = mul(&xim[a], &zim[a]); }
            ("sx" | "sxdg", 1) => { let a = qubits[0]; zim[a] = mul(&xim[a], &zim[a]); }
            ("cz", 2) => {
                let (a, b) = (qubits[0], qubits[1]);
                let (za, zb) = (zim[a].clone(), zim[b].clone());
                xim[a] = mul(&xim[a], &zb);
                xim[b] = mul(&xim[b], &za);
            }
            _ => {}
        }
    }
    zim
}

fn anti(p: &(Mask, Mask), q: &(Mask, Mask)) -> bool {
    (p.1.and(&q.0).count() + p.0.and(&q.1).count()) % 2 == 1
}

fn main() {
    println!("     n    t   nonzero    k = axes anticommuting with Q_S  ->  |coefficient|");
    for n in [8usize, 10, 12, 14] {
        let d = Dcs::scaled(n);
        let circuit = d.circuit();
        let axes = dcs::rotation_axes(&circuit).unwrap();
        let zim = z_images(&d.with_t(0).skeleton(), n);

        let st = Simulator::<C64>::new().run(&circuit).unwrap();
        let dim = 1usize << n;
        let mut p: Vec<f64> = (0..dim).map(|x| st.amplitude(x as u64).norm_sqr()).collect();
        fwht(&mut p);

        // class k -> set of |coefficient| values seen
        let mut by_k: BTreeMap<usize, BTreeMap<i64, usize>> = BTreeMap::new();
        let mut nz = 0;
        for s in 0..dim as u64 {
            let c = p[s as usize];
            if c.abs() <= 1e-9 { continue; }
            nz += 1;
            let mut q = (Mask::zero(), Mask::zero());
            for (bit, img) in zim.iter().enumerate().take(n) {
                if s >> bit & 1 == 1 {
                    q = (q.0.xor(&img.0), q.1.xor(&img.1));
                }
            }
            let k = axes.iter().filter(|a| anti(a, &q)).count();
            *by_k.entry(k).or_default().entry((c.abs() * 1e9).round() as i64).or_insert(0) += 1;
        }
        println!("  {n:>4} {:>4}  {nz:>7}", d.t_gates);
        for (k, vals) in &by_k {
            let list: Vec<String> = vals.iter().take(4)
                .map(|(v, c)| format!("{:.6}×{c}", *v as f64 / 1e9)).collect();
            println!("            k={k:<4} {:>5} coeffs, {:>3} distinct value(s): {}",
                vals.values().sum::<usize>(), vals.len(), list.join(", "));
        }
    }
}
