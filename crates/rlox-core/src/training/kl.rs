/// Adaptive KL penalty controller (Ziegler et al. 2019).
///
/// Adjusts a coefficient based on measured vs target KL divergence:
/// - If measured KL > 1.5 * target: multiply coefficient by 2
/// - If measured KL < target / 1.5: divide coefficient by 2
/// - Otherwise: leave coefficient unchanged
pub struct KLController {
    coefficient: f64,
    target_kl: f64,
}

impl KLController {
    pub fn new(init_coeff: f64, target_kl: f64) -> Self {
        Self {
            coefficient: init_coeff,
            target_kl,
        }
    }

    pub fn coefficient(&self) -> f64 {
        self.coefficient
    }

    pub fn update(&mut self, measured_kl: f64) {
        if measured_kl > 1.5 * self.target_kl {
            self.coefficient *= 2.0;
        } else if measured_kl < self.target_kl / 1.5 {
            self.coefficient /= 2.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kl_controller_initial_coefficient() {
        let kl = KLController::new(0.01, 0.02);
        assert_eq!(kl.coefficient(), 0.01);
    }

    #[test]
    fn kl_controller_increases_on_high_kl() {
        let mut kl = KLController::new(0.01, 0.02);
        kl.update(0.05); // measured KL >> target
        assert!(kl.coefficient() > 0.01);
    }

    #[test]
    fn kl_controller_decreases_on_low_kl() {
        let mut kl = KLController::new(0.01, 0.02);
        kl.update(0.005); // measured KL << target
        assert!(kl.coefficient() < 0.01);
    }

    #[test]
    fn kl_controller_stays_near_target() {
        let mut kl = KLController::new(0.01, 0.02);
        kl.update(0.02); // measured KL == target
        // Coefficient should remain exactly the same (within dead zone)
        assert!((kl.coefficient() - 0.01).abs() < 0.005);
    }

    #[test]
    fn kl_controller_has_floor() {
        let mut kl = KLController::new(0.01, 0.02);
        for _ in 0..100 {
            kl.update(0.0001); // very low KL
        }
        assert!(kl.coefficient() > 0.0); // never goes to zero
    }
}
