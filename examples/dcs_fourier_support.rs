use quantsim::dcs::{self, Dcs};
use quantsim::prelude::*;
use std::collections::HashSet;

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

/// GF(2) rank of a set of vectors.
fn rank(rows: &[u64]) -> usize {
    let mut piv: Vec<u64> = Vec::new();
    for &r in rows {
        let mut r = r;
        for &p in &piv {
            if r & (p & p.wrapping_neg()) != 0 {
                r ^= p;
            }
        }
        if r != 0 {
            piv.push(r);
            piv.sort_by_key(|x| x.trailing_zeros());
        }
    }
    piv.len()
}

fn main() {
    println!("     n    t   nonzero   subspace?  rank  span size   distinct |values|");
    for n in [6usize, 8, 10, 12, 14] {
        let d = Dcs::scaled(n);
        let st = Simulator::<C64>::new().run(&d.circuit()).unwrap();
        let dim = 1usize << n;
        let mut p: Vec<f64> = (0..dim)
            .map(|x| st.amplitude(x as u64).norm_sqr())
            .collect();
        fwht(&mut p);
        let nz: Vec<u64> = (0..dim as u64)
            .filter(|&s| p[s as usize].abs() > 1e-9)
            .collect();
        let set: HashSet<u64> = nz.iter().copied().collect();
        // Closed under XOR?
        let mut closed = true;
        'o: for &a in &nz {
            for &b in &nz {
                if !set.contains(&(a ^ b)) {
                    closed = false;
                    break 'o;
                }
            }
        }
        let r = rank(&nz);
        let mut vals: Vec<i64> = nz
            .iter()
            .map(|&s| (p[s as usize].abs() * 1e9).round() as i64)
            .collect();
        vals.sort_unstable();
        vals.dedup();
        println!(
            "  {n:>4} {:>4}  {:>8}   {:>8}  {r:>4}   {:>9}   {:>10}",
            d.t_gates,
            nz.len(),
            if closed { "YES" } else { "no" },
            1usize << r,
            vals.len()
        );
        // How much of the span is actually used?
        if !closed {
            let axes = dcs::rotation_axes(&d.circuit()).unwrap();
            let zparts: Vec<u64> = axes
                .iter()
                .map(|(_, z)| {
                    let mut v = 0u64;
                    for i in z.iter() {
                        v |= 1 << i;
                    }
                    v
                })
                .collect();
            println!(
                "        rank of the T axes' Z-parts: {}   (span {})",
                rank(&zparts),
                1usize << rank(&zparts)
            );
        }
    }
}
