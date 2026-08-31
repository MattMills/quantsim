//! Geometric qudits: the obstruction, the boundary it bounds, motion,
//! holonomy from motion, frame-in-frame, and the interior no face sees.
//!
//! Run with `cargo run --release --example geometric_qudits`.

use quantsim::backend::{CliffordStep, PauliString};
use quantsim::retro::{Code, SurfaceCode, ToricCode};
use quantsim::stitch::Volume;
use quantsim::volqudit::*;

fn ps(x: u64, z: u64) -> PauliString {
    PauliString {
        x,
        z,
        negative: false,
    }
}

fn rule(t: &str) {
    println!("\n══ {t} ══\n");
}

fn show(p: PauliString, n: usize) -> String {
    (0..n)
        .map(|q| match (p.x >> q & 1 == 1, p.z >> q & 1 == 1) {
            (true, true) => 'Y',
            (true, false) => 'X',
            (false, true) => 'Z',
            _ => '.',
        })
        .collect()
}

fn main() -> quantsim::Result<()> {
    rule("1. the qudit is the obstruction, and its levels are the geometry's");
    println!("  frame              n     r   boundary   h   levels        admissible   End");
    for (label, n, gens) in [
        ("free (no frame)", 4usize, vec![]),
        ("GHZ  ZZ., .ZZ", 3, vec![ps(0, 0b011), ps(0, 0b110)]),
        ("toric L=2", 8, ToricCode::new(2, 0)?.generators()),
        ("toric L=3", 18, ToricCode::new(3, 0)?.generators()),
        ("surface d=3", 9, SurfaceCode::new(3, 0)?.generators()),
        ("surface d=5", 25, SurfaceCode::new(5, 0)?.generators()),
    ] {
        let q = VolQudit::new(n, Volume::span(n, &gens)?)?;
        let c = q.operative_cost();
        println!(
            "  {label:<17} {n:>2}  {:>4}   {:>8}  {:>2}   {:>6}   {:>7} bits   {:>7}",
            q.frame().rank(),
            q.boundary().rank(),
            q.logical_rank(),
            q.levels(),
            c.admissible_bits,
            c.full_end_params
        );
    }
    println!(
        "\n  Nothing declared a level count. `V` is an isotropic flat — a constraint —\n  \
         and `V^perp` is the boundary it bounds, rank 2n - r, always containing V.\n  \
         `V^perp/V` carries a non-degenerate form of rank 2h, so levels = 2^h.\n  \
         Naming an admissible operation costs 2h BITS; a general element of End on\n  \
         the same level space costs 4^h COMPLEX NUMBERS. That gap is A_V < End(V),\n  \
         forced by the geometry rather than imposed on it."
    );

    rule("2. the conjugate pairs are found, not declared");
    let ghz = VolQudit::new(3, Volume::span(3, &[ps(0, 0b011), ps(0, 0b110)])?)?;
    println!("  frame:");
    for g in ghz.frame().basis() {
        println!("    {}", show(g, 3));
    }
    println!(
        "  boundary rank {} ; conjugate pairs from its normal form:",
        ghz.boundary().rank()
    );
    for (i, &(x, z)) in ghz.logical_pairs().iter().enumerate() {
        println!(
            "    pair {i}:  e = {}   f = {}   anticommute {}",
            show(x, 3),
            show(z, 3),
            !x.commutes_with(z)
        );
    }

    rule("3. motion: the frame moves, the qudit does not change");
    let sig = ghz.signature();
    for steps in [
        vec![CliffordStep::H(0)],
        vec![CliffordStep::Cx(0, 1)],
        vec![
            CliffordStep::S(2),
            CliffordStep::Cx(1, 2),
            CliffordStep::H(1),
        ],
    ] {
        let m = ghz.transport(&steps)?;
        println!(
            "  transport {:<44} moved {}   signature preserved {}",
            format!("{steps:?}"),
            !m.frame().is_same(ghz.frame()),
            m.signature() == sig
        );
    }

    rule("4. holonomy: bring it back and it has been acted on");
    let free = VolQudit::free(1)?;
    println!("  a free qubit — every Clifford word is a loop, so the search sees all of them:");
    for h in free.holonomy_search(&[CliffordStep::H(0), CliffordStep::S(0)], 3) {
        let m = h.matrix();
        println!(
            "    [{}{} ; {}{}]   curvature {}   order {:?}",
            u8::from(m[0][0]),
            u8::from(m[0][1]),
            u8::from(m[1][0]),
            u8::from(m[1][1]),
            h.curvature(),
            h.order(12)
        );
    }
    println!("  — orders 2 and 3: the whole of SL(2,F2) = S3, from motion alone.\n");
    let gens = [
        CliffordStep::Cx(0, 1),
        CliffordStep::Cx(1, 2),
        CliffordStep::Cx(1, 0),
        CliffordStep::S(0),
        CliffordStep::S(1),
        CliffordStep::S(2),
    ];
    let found = ghz.holonomy_search(&gens, 3);
    println!("  the GHZ qudit is fixed by far fewer words. Loops with curvature, depth 3:");
    for h in &found {
        println!("    curvature {}   order {:?}", h.curvature(), h.order(16));
    }
    println!(
        "\n  A loop returns the frame to itself. What the frame BOUNDS need not come\n  \
         back — the induced action on V^perp/V is a logical operation obtained by\n  \
         moving the qudit around the ambient and returning it. Flat loops report\n  \
         curvature 0 and are excluded; an open path is refused, since holonomy of\n  \
         a path that does not close is not defined."
    );

    rule("5. frame in frame");
    let outer = VolQudit::free(6)?;
    let inner = outer.nest(Volume::span(6, &[ps(0, 0b000011), ps(0, 0b001100)])?)?;
    println!(
        "  outer: ambient {} qubits, h = {}, levels {}",
        outer.ambient(),
        outer.logical_rank(),
        outer.levels()
    );
    println!(
        "  inner: ambient IS the outer level space, h = {}, levels {}",
        inner.logical_rank(),
        inner.levels()
    );
    println!(
        "  inner frame costs {} bits — O(h), not O(n): the outer geometry already paid.",
        inner.operative_cost().frame_bits
    );

    rule("6. promotion: the interface, and nothing else");
    let a = VolQudit::new(9, Volume::span(9, &SurfaceCode::new(3, 0)?.generators())?)?;
    let b = VolQudit::new(
        9,
        Volume::span(9, &(0..8).map(|q| ps(0, 1 << q)).collect::<Vec<_>>())?,
    )?;
    println!(
        "  surface d=3 frame rank {} vs a plain Z-frame rank {}",
        a.frame().rank(),
        b.frame().rank()
    );
    println!(
        "  same frame? {}   same signature? {}   same atom? {}",
        a.frame().is_same(b.frame()),
        a.signature() == b.signature(),
        a.promote() == b.promote()
    );
    println!(
        "\n  Interface sufficiency is a conjecture in the theory. This does not prove\n  \
         it — it makes it checkable at this scale: two structurally different closed\n  \
         vols with equal sigma promote to the same atom, and an outer context that\n  \
         only reads the atom cannot tell them apart."
    );

    rule("7. the interior: what no face can see");
    println!("  compound              h   faces        skeleton   interior   brunnian");
    for (label, q, slots) in [
        (
            "3 free qubits",
            VolQudit::free(3)?,
            vec![(0, 1), (1, 1), (2, 1)],
        ),
        ("GHZ frame", ghz.clone(), vec![(0, 1), (1, 1), (2, 1)]),
        (
            "toric L=2, halves",
            VolQudit::new(8, Volume::span(8, &ToricCode::new(2, 0)?.generators())?)?,
            vec![(0, 4), (4, 4)],
        ),
    ] {
        let c = CompoundQudit::new(q, &slots)?;
        let faces: Vec<usize> = (0..slots.len()).map(|i| c.face_rank(i).unwrap()).collect();
        println!(
            "  {label:<20} {:>2}   {:<12} {:>8}   {:>8}   {}",
            c.qudit().logical_rank(),
            format!("{faces:?}"),
            c.skeleton_rank()?,
            c.interior_rank()?,
            c.is_brunnian()?
        );
    }
    println!(
        "\n  The control is the first row: independent qudits side by side have every\n  \
         logical class representable on a single slot, so every face sees it, the\n  \
         skeleton is all of V^perp/V, and the vol is EMPTY. The GHZ frame keeps one\n  \
         dimension that no face can see: one member of its conjugate pair can be\n  \
         pushed onto a single qubit by multiplying in a stabilizer, and its partner\n  \
         XXX cannot be pushed off any qubit at all. That surviving dimension is the\n  \
         interior, and it is asked of the COSET rather than of a spelling, so it does\n  \
         not depend on which representative the basis happened to hold."
    );

    rule("8. the whole thing is polynomial");
    let n = 40;
    let t0 = std::time::Instant::now();
    let big = VolQudit::new(
        n,
        Volume::span(n, &(0..4).map(|q| ps(0, 1 << q)).collect::<Vec<_>>())?,
    )?;
    let c = CompoundQudit::new(big.clone(), &[(0, 20), (20, 20)])?;
    let interior = c.interior_rank()?;
    println!(
        "  ambient {n} qubits, frame rank {}, h = {}, levels 2^{} = {}",
        big.frame().rank(),
        big.logical_rank(),
        big.logical_rank(),
        big.levels()
    );
    println!(
        "  frame {} bits, interior rank {interior}, computed in {:?}",
        big.operative_cost().frame_bits,
        t0.elapsed()
    );
    println!(
        "\n  Every quantity above — levels, boundary rank, conjugate pairs, holonomy,\n  \
         interior — is F2 mask algebra at O(rank . n). A 2^36-level qudit is a few\n  \
         hundred bits and microseconds, and nothing anywhere materializes 2^n."
    );
    Ok(())
}
