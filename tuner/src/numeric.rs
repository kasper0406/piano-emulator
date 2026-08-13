//! The small numerical toolbox the estimators share: dense least squares on a
//! handful of unknowns, a derivative-free minimizer for the nonlinear fits, an
//! isotonic regression for the monotonicity constraints, and robust statistics.
//!
//! Everything here is deliberately tiny and dependency-free. The problems the
//! estimators pose are two to four unknowns over a few hundred points, which is
//! far below the size at which a linear-algebra crate starts paying for itself,
//! and the fits are all cases where knowing exactly what the solver does
//! matters more than how fast it does it.

/// Solves the dense `n x n` system `a x = b` by Gaussian elimination with
/// partial pivoting. `a` is row-major and both inputs are consumed. Returns
/// `None` when the matrix is singular to working precision, which the callers
/// read as "this fit is not determined by the data" rather than as an error.
pub fn solve_in_place(a: &mut [f64], b: &mut [f64], n: usize) -> Option<Vec<f64>> {
    debug_assert_eq!(a.len(), n * n);
    debug_assert_eq!(b.len(), n);
    for column in 0..n {
        let (mut pivot, mut best) = (column, a[column * n + column].abs());
        for row in column + 1..n {
            let value = a[row * n + column].abs();
            if value > best {
                pivot = row;
                best = value;
            }
        }
        if best <= 1e-300 {
            return None;
        }
        if pivot != column {
            for c in 0..n {
                a.swap(column * n + c, pivot * n + c);
            }
            b.swap(column, pivot);
        }
        let diagonal = a[column * n + column];
        for row in column + 1..n {
            let factor = a[row * n + column] / diagonal;
            if factor == 0.0 {
                continue;
            }
            for c in column..n {
                a[row * n + c] -= factor * a[column * n + c];
            }
            b[row] -= factor * b[column];
        }
    }
    let mut x = vec![0.0; n];
    for row in (0..n).rev() {
        let mut sum = b[row];
        for c in row + 1..n {
            sum -= a[row * n + c] * x[c];
        }
        x[row] = sum / a[row * n + row];
    }
    if x.iter().all(|v| v.is_finite()) {
        Some(x)
    } else {
        None
    }
}

/// Weighted least squares against arbitrary basis functions: minimizes
/// `sum_i w_i (y_i - sum_j c_j phi_j(i))^2` over the coefficients `c`.
///
/// `basis` is row-major, one row of `terms` values per sample.
pub fn weighted_least_squares(
    basis: &[f64],
    y: &[f64],
    weights: &[f64],
    terms: usize,
) -> Option<Vec<f64>> {
    if terms == 0 || y.len() != weights.len() || basis.len() != y.len() * terms {
        return None;
    }
    let mut normal = vec![0.0; terms * terms];
    let mut rhs = vec![0.0; terms];
    for (sample, (&value, &weight)) in y.iter().zip(weights).enumerate() {
        if weight <= 0.0 || !value.is_finite() {
            continue;
        }
        let row = &basis[sample * terms..(sample + 1) * terms];
        for i in 0..terms {
            rhs[i] += weight * row[i] * value;
            for j in 0..terms {
                normal[i * terms + j] += weight * row[i] * row[j];
            }
        }
    }
    solve_in_place(&mut normal, &mut rhs, terms)
}

/// Weighted polynomial fit of `y` against `x`, returning coefficients in
/// ascending powers.
pub fn weighted_polyfit(x: &[f64], y: &[f64], weights: &[f64], degree: usize) -> Option<Vec<f64>> {
    let terms = degree + 1;
    let mut basis = vec![0.0; x.len() * terms];
    for (sample, &xi) in x.iter().enumerate() {
        let mut power = 1.0;
        for term in 0..terms {
            basis[sample * terms + term] = power;
            power *= xi;
        }
    }
    weighted_least_squares(&basis, y, weights, terms)
}

/// Evaluates a polynomial given in ascending powers.
pub fn poly_eval(coefficients: &[f64], x: f64) -> f64 {
    coefficients.iter().rev().fold(0.0, |acc, &c| acc * x + c)
}

/// The nondecreasing sequence closest to `values` in weighted least squares,
/// by pool adjacent violators.
///
/// This is how every monotonicity constraint in the crate is enforced: fit the
/// values freely, then project onto the monotone cone. The projection is exact
/// (PAVA is the solution of the constrained problem, not an approximation of
/// it) and costs one pass.
pub fn isotonic(values: &[f64], weights: &[f64]) -> Vec<f64> {
    let mut level: Vec<f64> = Vec::with_capacity(values.len());
    let mut mass: Vec<f64> = Vec::with_capacity(values.len());
    let mut count: Vec<usize> = Vec::with_capacity(values.len());
    for (&value, &weight) in values.iter().zip(weights) {
        let weight = if weight > 0.0 { weight } else { f64::MIN_POSITIVE };
        level.push(value);
        mass.push(weight);
        count.push(1);
        // Merge back while the new block is below its left neighbour.
        while level.len() > 1 && level[level.len() - 2] > level[level.len() - 1] {
            let (v, w, c) = (level.pop().unwrap(), mass.pop().unwrap(), count.pop().unwrap());
            let last = level.len() - 1;
            let total = mass[last] + w;
            level[last] = (level[last] * mass[last] + v * w) / total;
            mass[last] = total;
            count[last] += c;
        }
    }
    let mut out = Vec::with_capacity(values.len());
    for (value, repeats) in level.into_iter().zip(count) {
        out.extend(std::iter::repeat_n(value, repeats));
    }
    out
}

/// Median of a sample. `None` for an empty one.
pub fn median(values: &[f64]) -> Option<f64> {
    if values.is_empty() {
        return None;
    }
    let mut v = values.to_vec();
    v.sort_by(f64::total_cmp);
    let mid = v.len() / 2;
    Some(if v.len() % 2 == 0 {
        0.5 * (v[mid - 1] + v[mid])
    } else {
        v[mid]
    })
}

/// Weighted mean. `None` when the weights sum to zero.
pub fn weighted_mean(values: &[f64], weights: &[f64]) -> Option<f64> {
    let (mut num, mut den) = (0.0, 0.0);
    for (&v, &w) in values.iter().zip(weights) {
        if w > 0.0 && v.is_finite() {
            num += w * v;
            den += w;
        }
    }
    if den > 0.0 {
        Some(num / den)
    } else {
        None
    }
}

/// Result of a minimization.
#[derive(Clone, Debug)]
pub struct Minimum {
    pub point: Vec<f64>,
    pub value: f64,
    pub evaluations: usize,
    /// Whether the simplex collapsed inside the tolerance before the budget ran
    /// out. A fit that reports `false` is not necessarily wrong, but it has not
    /// proved anything either.
    pub converged: bool,
}

/// Nelder-Mead simplex minimization.
///
/// Derivative-free by choice: the objectives here run an ODE (the hammer's
/// contact) or a nested linear solve (every variable-projection fit), so an
/// analytic gradient does not exist in closed form and a numerical one costs
/// what the simplex costs anyway. Every caller works in log-parameters, which
/// both keeps the positive quantities positive and makes one absolute step size
/// meaningful across parameters of wildly different scale.
#[derive(Clone, Copy, Debug)]
pub struct NelderMead {
    pub max_evaluations: usize,
    /// Convergence test on the spread of the simplex's vertices, in parameter
    /// units, and on the spread of its values, relative to the best.
    pub tolerance: f64,
    pub initial_step: f64,
}

impl Default for NelderMead {
    fn default() -> Self {
        Self {
            max_evaluations: 2_000,
            tolerance: 1e-8,
            initial_step: 0.25,
        }
    }
}

impl NelderMead {
    pub fn minimize<F: FnMut(&[f64]) -> f64>(&self, start: &[f64], mut objective: F) -> Minimum {
        let n = start.len();
        assert!(n > 0, "nothing to minimize over");
        let mut evaluations = 0;
        let mut evaluate = |point: &[f64], evaluations: &mut usize| {
            *evaluations += 1;
            let value = objective(point);
            if value.is_finite() {
                value
            } else {
                f64::MAX
            }
        };

        let mut simplex: Vec<(Vec<f64>, f64)> = Vec::with_capacity(n + 1);
        let value = evaluate(start, &mut evaluations);
        simplex.push((start.to_vec(), value));
        for i in 0..n {
            let mut point = start.to_vec();
            point[i] += self.initial_step;
            let value = evaluate(&point, &mut evaluations);
            simplex.push((point, value));
        }

        let mut converged = false;
        while evaluations < self.max_evaluations {
            simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
            let (best, worst) = (&simplex[0], &simplex[n]);
            let spread = simplex
                .iter()
                .map(|(p, _)| {
                    p.iter()
                        .zip(&best.0)
                        .map(|(a, b)| (a - b).abs())
                        .fold(0.0, f64::max)
                })
                .fold(0.0, f64::max);
            let value_spread = (worst.1 - best.1).abs();
            if spread < self.tolerance
                && value_spread <= self.tolerance * (1.0 + best.1.abs())
            {
                converged = true;
                break;
            }

            // Centroid of everything but the worst vertex.
            let mut centroid = vec![0.0; n];
            for (point, _) in &simplex[..n] {
                for (c, p) in centroid.iter_mut().zip(point) {
                    *c += p / n as f64;
                }
            }
            let step = |factor: f64| -> Vec<f64> {
                centroid
                    .iter()
                    .zip(&simplex[n].0)
                    .map(|(c, w)| c + factor * (c - w))
                    .collect()
            };

            let reflected = step(1.0);
            let reflected_value = evaluate(&reflected, &mut evaluations);
            if reflected_value < simplex[0].1 {
                let expanded = step(2.0);
                let expanded_value = evaluate(&expanded, &mut evaluations);
                simplex[n] = if expanded_value < reflected_value {
                    (expanded, expanded_value)
                } else {
                    (reflected, reflected_value)
                };
            } else if reflected_value < simplex[n - 1].1 {
                simplex[n] = (reflected, reflected_value);
            } else {
                let inside = reflected_value >= simplex[n].1;
                let contracted = step(if inside { -0.5 } else { 0.5 });
                let contracted_value = evaluate(&contracted, &mut evaluations);
                if contracted_value < simplex[n].1.min(reflected_value) {
                    simplex[n] = (contracted, contracted_value);
                } else {
                    // Shrink towards the best vertex.
                    let best = simplex[0].0.clone();
                    for vertex in simplex[1..].iter_mut() {
                        for (p, b) in vertex.0.iter_mut().zip(&best) {
                            *p = b + 0.5 * (*p - b);
                        }
                        vertex.1 = evaluate(&vertex.0, &mut evaluations);
                    }
                }
            }
        }

        simplex.sort_by(|a, b| a.1.total_cmp(&b.1));
        let (point, value) = simplex.swap_remove(0);
        Minimum {
            point,
            value,
            evaluations,
            converged,
        }
    }
}

/// Golden-section search for the minimum of a unimodal function on `[lo, hi]`.
/// Returns the location and value.
pub fn golden_section<F: FnMut(f64) -> f64>(
    lo: f64,
    hi: f64,
    iterations: usize,
    mut f: F,
) -> (f64, f64) {
    const INVERSE_PHI: f64 = 0.618_033_988_749_895;
    let (mut a, mut b) = (lo, hi);
    let mut c = b - INVERSE_PHI * (b - a);
    let mut d = a + INVERSE_PHI * (b - a);
    let (mut fc, mut fd) = (f(c), f(d));
    for _ in 0..iterations {
        if fc < fd {
            b = d;
            d = c;
            fd = fc;
            c = b - INVERSE_PHI * (b - a);
            fc = f(c);
        } else {
            a = c;
            c = d;
            fc = fd;
            d = a + INVERSE_PHI * (b - a);
            fd = f(d);
        }
    }
    if fc < fd {
        (c, fc)
    } else {
        (d, fd)
    }
}

/// Sub-sample location of the extremum of the parabola through
/// `(-1, left), (0, centre), (1, right)`, as an offset in `[-1, 1]` from the
/// centre. Zero when the three points are collinear.
pub fn parabolic_offset(left: f64, centre: f64, right: f64) -> f64 {
    let denominator = left - 2.0 * centre + right;
    if denominator.abs() < 1e-300 {
        return 0.0;
    }
    (0.5 * (left - right) / denominator).clamp(-1.0, 1.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_dense_solver_inverts_a_small_system() {
        // [2 1; 1 3] x = [5; 10]  ->  x = [1; 3]
        let mut a = vec![2.0, 1.0, 1.0, 3.0];
        let mut b = vec![5.0, 10.0];
        let x = solve_in_place(&mut a, &mut b, 2).unwrap();
        assert!((x[0] - 1.0).abs() < 1e-12 && (x[1] - 3.0).abs() < 1e-12);

        let mut singular = vec![1.0, 2.0, 2.0, 4.0];
        let mut rhs = vec![1.0, 2.0];
        assert!(solve_in_place(&mut singular, &mut rhs, 2).is_none());
    }

    #[test]
    fn the_polynomial_fit_recovers_its_own_coefficients() {
        let truth = [0.5, -1.25, 0.75];
        let x: Vec<f64> = (0..20).map(|i| f64::from(i) * 0.1).collect();
        let y: Vec<f64> = x.iter().map(|&x| poly_eval(&truth, x)).collect();
        let w = vec![1.0; x.len()];
        let fit = weighted_polyfit(&x, &y, &w, 2).unwrap();
        for (a, b) in fit.iter().zip(truth) {
            assert!((a - b).abs() < 1e-9, "{fit:?}");
        }
    }

    #[test]
    fn isotonic_regression_is_the_monotone_projection() {
        let values = [1.0, 3.0, 2.0, 4.0];
        let weights = [1.0; 4];
        let fit = isotonic(&values, &weights);
        // The violating pair is pooled to its mean and nothing else moves.
        assert_eq!(fit, vec![1.0, 2.5, 2.5, 4.0]);
        assert!(fit.windows(2).all(|w| w[0] <= w[1]));

        // Already monotone data is left alone; reversed data collapses to the
        // weighted mean.
        assert_eq!(isotonic(&[1.0, 2.0, 3.0], &weights[..3]), vec![1.0, 2.0, 3.0]);
        let flat = isotonic(&[3.0, 2.0, 1.0], &weights[..3]);
        assert!(flat.iter().all(|&v| (v - 2.0).abs() < 1e-12));
    }

    #[test]
    fn nelder_mead_finds_the_rosenbrock_valley() {
        let solver = NelderMead {
            max_evaluations: 5_000,
            tolerance: 1e-10,
            initial_step: 0.5,
        };
        let minimum = solver.minimize(&[-1.2, 1.0], |p| {
            let (x, y) = (p[0], p[1]);
            (1.0 - x).powi(2) + 100.0 * (y - x * x).powi(2)
        });
        assert!(minimum.converged, "{minimum:?}");
        assert!((minimum.point[0] - 1.0).abs() < 1e-4, "{minimum:?}");
        assert!((minimum.point[1] - 1.0).abs() < 1e-4, "{minimum:?}");
    }

    #[test]
    fn golden_section_finds_a_smooth_minimum() {
        let (x, value) = golden_section(-2.0, 5.0, 80, |x| (x - 1.234).powi(2) + 0.5);
        // A quadratic minimum can only ever be located to about the square root
        // of the machine epsilon: a step of 1e-8 changes the value by 1e-16,
        // which is below the resolution of the value itself.
        assert!((x - 1.234).abs() < 1e-7, "{x}");
        assert!((value - 0.5).abs() < 1e-15);
    }

    #[test]
    fn the_parabolic_offset_lands_on_the_true_vertex() {
        // A parabola with its vertex 0.25 bins right of the centre sample.
        let f = |x: f64| (x - 0.25).powi(2);
        let offset = parabolic_offset(f(-1.0), f(0.0), f(1.0));
        assert!((offset - 0.25).abs() < 1e-12, "{offset}");
        assert_eq!(parabolic_offset(1.0, 1.0, 1.0), 0.0);
    }
}
