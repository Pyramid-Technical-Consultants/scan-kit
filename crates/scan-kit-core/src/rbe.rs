//! Proton RBE: the clinical constant 1.1, and variable models of LETd, dose per
//! fraction, and (α/β)x.
//!
//! The linear-quadratic models give RBEmax (the RBE as the dose goes to 0) and
//! RBEmin (as it grows); at a proton dose `d` per fraction
//!
//! ```text
//! RBE = (sqrt((α/β)² + 4 d (α/β) RBEmax + 4 d² RBEmin²) − α/β) / (2 d)
//! ```
//!
//! LETd is in water at unit density, from primary and secondary protons: what
//! the models were fitted to and what the EPTN consensus (Phys Imaging Radiat
//! Oncol 2024) asks centres to report.

/// The clinical constant for physical doses and Gy(RBE) conversions.
pub const RBE_CONSTANT: f64 = 1.1;

/// Gy: the Dutch centres' McNamara reporting value (Radiother Oncol 202, 110653).
pub const DEFAULT_ALPHA_BETA: f64 = 2.0;

/// Unkelbach slope, µm/keV.
pub const UNKELBACH_C: f64 = 0.04;

/// Variable-RBE model, or the clinical constant.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum RbeModel {
    Constant,
    McNamara,
    Wedenberg,
    Carabe,
    Unkelbach,
}

impl RbeModel {
    /// Model from the control key (`"mcnamara"`, …); `"1"`/`"1.1"` are the constant.
    pub fn parse(name: &str) -> Self {
        match name {
            "mcnamara" => Self::McNamara,
            "wedenberg" => Self::Wedenberg,
            "carabe" => Self::Carabe,
            "unkelbach" => Self::Unkelbach,
            _ => Self::Constant,
        }
    }

    /// The linear-quadratic models take an (α/β)x input.
    pub fn uses_alpha_beta(self) -> bool {
        matches!(self, Self::McNamara | Self::Wedenberg | Self::Carabe)
    }

    /// Every variable model needs LETd scored by the transport.
    pub fn uses_let(self) -> bool {
        self != Self::Constant
    }
}

/// (RBEmax, RBEmin) at LETd `letd` (keV/µm) for (α/β)x in Gy.
///
/// `None` for the non-linear-quadratic models, matching the Python `ValueError`.
pub fn rbe_limits(model: RbeModel, letd: f64, alpha_beta: f64) -> Option<(f64, f64)> {
    let ab = alpha_beta;
    let (hi, lo) = match model {
        // Fit values as tabulated with their standard errors.
        RbeModel::McNamara => (
            0.99064 + 0.35605 / ab * letd,
            1.1012 - 0.0038703 * ab.sqrt() * letd,
        ),
        RbeModel::Wedenberg => (1.0 + 0.434 / ab * letd, 1.0),
        RbeModel::Carabe => {
            let r = 2.686 / ab * letd;
            (0.843 + 0.154 * r, 1.09 + 0.006 * r)
        }
        _ => return None,
    };
    Some((hi, lo))
}

/// RBE per voxel: `dose` physical Gy per fraction, `letd` LETd keV/µm in water.
pub fn rbe(model: RbeModel, dose: f64, letd: f64, alpha_beta: f64) -> f64 {
    match model {
        RbeModel::Constant => RBE_CONSTANT,
        RbeModel::Unkelbach => 1.0 + UNKELBACH_C * letd.max(0.0),
        _ => {
            let Some((hi, lo)) = rbe_limits(model, letd.max(0.0), alpha_beta) else {
                return RBE_CONSTANT;
            };
            let d = dose.max(1e-6);
            let r = ((alpha_beta * alpha_beta + 4.0 * d * alpha_beta * hi + 4.0 * d * d * lo * lo)
                .sqrt()
                - alpha_beta)
                / (2.0 * d);
            if dose > 1e-6 {
                r
            } else {
                hi
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const LQ: [RbeModel; 3] = [RbeModel::McNamara, RbeModel::Wedenberg, RbeModel::Carabe];

    // The LQ solve gives back the photon dose of equal effect: alpha_x·dx +
    // beta_x·dx² == RBEmax·alpha_x·d + RBEmin²·beta_x·d².
    #[test]
    fn rbe_gives_the_photon_dose_of_equal_effect() {
        let beta_x = 0.05;
        let doses = [0.5, 1.8, 2.0, 5.0, 8.0];
        let lets = [0.5, 2.0, 4.0, 8.0, 15.0];
        for model in LQ {
            for ab in [2.0, 3.0, 10.0] {
                let alpha_x = ab * beta_x;
                for (d, letd) in doses.iter().zip(&lets) {
                    let (hi, lo) = rbe_limits(model, *letd, ab).unwrap();
                    let proton = hi * alpha_x * d + lo * lo * beta_x * d * d;
                    let dx = rbe(model, *d, *letd, ab) * d;
                    let got = alpha_x * dx + beta_x * dx * dx;
                    assert!(
                        (got - proton).abs() <= 1e-9 * proton.abs().max(1.0),
                        "{model:?} d={d} let={letd} ab={ab}: {got} != {proton}"
                    );
                }
            }
        }
    }

    #[test]
    fn rbe_limits_and_trends() {
        for model in LQ {
            for k in 1..=12 {
                let letd = f64::from(k);
                // dose -> 0 gives RBEmax.
                let (hi, _) = rbe_limits(model, letd, 2.0).unwrap();
                assert!((rbe(model, 0.0, letd, 2.0) - hi).abs() < 1e-12);
            }
            // More LET, more effect.
            let r: Vec<f64> = (1..=12)
                .map(|k| rbe(model, 2.0, f64::from(k), 2.0))
                .collect();
            assert!(r.windows(2).all(|pair| pair[1] > pair[0]), "{model:?}");
            // Late-responding tissue (higher (α/β)x) gains less.
            for k in 1..=12 {
                let letd = f64::from(k);
                assert!(rbe(model, 2.0, letd, 10.0) < rbe(model, 2.0, letd, 2.0));
            }
        }
        assert_eq!(rbe(RbeModel::Constant, 1.0, 5.0, 2.0), 1.1);
        assert!((rbe(RbeModel::Unkelbach, 2.0, 2.5, 2.0) - 1.1).abs() < 1e-12);
        // McNamara in a typical target (2 Gy, LETd 2 keV/µm, (α/β)x 10 Gy):
        // close to the clinical 1.1.
        assert!((rbe(RbeModel::McNamara, 2.0, 2.0, 10.0) - 1.0663).abs() < 1e-3);
        assert!(rbe_limits(RbeModel::Constant, 1.0, 2.0).is_none());
        assert!(rbe_limits(RbeModel::Unkelbach, 1.0, 2.0).is_none());
    }
}
