//! Every error variant renders a useful message — the Display surface is
//! part of the API contract (these strings end up in research logs).

use quantsim::prelude::*;

#[test]
fn every_variant_formats_informatively() {
    let cases: Vec<(Error, &[&str])> = vec![
        (Error::UnknownGate("zz9".into()), &["unknown gate", "zz9"]),
        (
            Error::UnknownBackend("mps".into()),
            &["unknown backend", "mps"],
        ),
        (
            Error::DuplicateGate("h".into()),
            &["h", "already registered"],
        ),
        (
            Error::DuplicateBackend("dense".into()),
            &["dense", "already registered"],
        ),
        (
            Error::QubitOutOfRange {
                qubit: 7,
                num_qubits: 3,
            },
            &["qubit 7", "width 3"],
        ),
        (
            Error::DuplicateQubits { qubits: vec![1, 1] },
            &["duplicate qubit", "[1, 1]"],
        ),
        (
            Error::ArityMismatch {
                gate: "cx".into(),
                expected: 2,
                got: 3,
            },
            &["cx", "2 qubit(s)", "got 3"],
        ),
        (
            Error::ParamCountMismatch {
                gate: "rx".into(),
                expected: 1,
                got: 0,
            },
            &["rx", "1 parameter(s)", "got 0"],
        ),
        (
            Error::BadDimension {
                expected: 4,
                got: 2,
            },
            &["dimension 2", "expected 4"],
        ),
        (Error::NonPowerOfTwoDim(3), &["dimension 3", "power of two"]),
        (
            Error::NotUnitary {
                label: "mystery".into(),
                deviation: 0.5,
            },
            &["mystery", "not unitary", "5.000e-1"],
        ),
        (
            Error::UnsupportedForAlgebra {
                gate: "s".into(),
                algebra: "R".into(),
            },
            &["'s'", "algebra R"],
        ),
        (
            Error::TooManyQubits {
                requested: 40,
                max: 32,
            },
            &["40 qubits", "at most 32"],
        ),
        (
            Error::WidthMismatch {
                circuit: 2,
                backend: 3,
            },
            &["circuit width 2", "backend width 3"],
        ),
        (
            Error::InvalidState("negative weight".into()),
            &["invalid state", "negative weight"],
        ),
        (
            Error::OutOfMemory {
                requested: 1 << 40,
                available: 1 << 30,
                what: "dense state (36 qubits)".into(),
            },
            &["out of memory", "dense state (36 qubits)", "measured"],
        ),
        (
            Error::Timeout {
                budget_ms: 1000,
                elapsed_ms: 1417,
            },
            &["time budget exceeded", "1417", "1000"],
        ),
    ];
    for (err, fragments) in cases {
        let rendered = err.to_string();
        for fragment in fragments {
            assert!(
                rendered.contains(fragment),
                "{err:?} rendered as {rendered:?}, missing {fragment:?}"
            );
        }
        // The std::error::Error impl is wired up (no sources; Debug works).
        let dynamic: &dyn std::error::Error = &err;
        assert!(dynamic.source().is_none());
        assert!(!format!("{err:?}").is_empty());
    }
}

#[test]
fn errors_propagate_through_result_alias() {
    fn fails() -> Result<()> {
        Err(Error::UnknownGate("nope".into()))
    }
    let message = fails().unwrap_err().to_string();
    assert!(message.contains("nope"));
}
