use nalgebra::{Matrix3x4, Vector3, Vector2, Vector4, DMatrix, DVector};
use nalgebra::linalg::SVD;
use nalgebra::base::VecStorage;
use nalgebra::dimension::{Dyn, U1};
use ndarray::Array2;
use levenberg_marquardt::{LeastSquaresProblem, LevenbergMarquardt};
use rand::rng;


fn project_point(P: Matrix3x4<f64>, x: Vector3<f64>) -> Vector2<f64> {
    let x_h = Vector4::new(x[0], x[1], x[2], 1.0);
    let x_proj = P * x_h;
    Vector2::new(x_proj[0] / x_proj[2], x_proj[1] / x_proj[2])
}

fn dlt_triangulation(P_list: &[Matrix3x4<f64>], pixels: &[Vector2<f64>]) -> Vector3<f64> {
    let n = P_list.len();
    let mut A = DMatrix::<f64>::zeros(2 * n, 4);
    for i in 0..n {
        let P = P_list[i];
        let pixel = pixels[i];
        let u = pixel[0];
        let v = pixel[1];
        for j in 0..4 {
            A[(2 * i, j)] = u * P[(2, j)] - P[(0, j)];
            A[(2 * i + 1, j)] = v * P[(2, j)] - P[(1, j)];
        }
    }
    //println!("This is A: {:?}", A);
    //println!("Size of A: {:?} x {:?}", A.nrows(), A.ncols());
    let svd = SVD::new(A, false, true);
    //println!("This is svd: {:?}", svd);
    let v_t = svd.v_t.expect("SVD with compute_v=true");
    let v = v_t.transpose();
    let X_h = v.column(v.ncols() - 1);
    Vector3::new(X_h[0] / X_h[3], X_h[1] / X_h[3], X_h[2] / X_h[3])
}

struct TrajectoryProblem {
    params: DVector<f64>,
    P_list: Vec<Matrix3x4<f64>>,
    pixels: Vec<Vec<Vector2<f64>>>,
    n_timesteps: usize,
    omega_phys: f64,
    dt: f64,
    g: Vector3<f64>,
    drag: f64,
    pixel_sigma: f64,
    physics_sigma: f64,
}

impl TrajectoryProblem {
    fn n_residuals(&self) -> usize {
        let n_cams = self.P_list.len();
        self.n_timesteps * n_cams * 2 + self.n_timesteps.saturating_sub(2) * 3
    }

    fn numerical_jacobian(&self, epsilon: f64) -> DMatrix<f64> {
        let n = self.params.len();
        let m = self.n_residuals();
        let r0 = self.residuals().unwrap_or_else(|| DVector::zeros(m));
        let mut J = DMatrix::zeros(m, n);
        let mut params_plus = self.params.clone();
        let mut params_minus = self.params.clone();
        for j in 0..n {
            let xj = self.params[j];
            params_plus[j] = xj + epsilon;
            params_minus[j] = xj - epsilon;
            let mut prob_plus = TrajectoryProblem {
                params: params_plus.clone(),
                P_list: self.P_list.clone(),
                pixels: self.pixels.clone(),
                n_timesteps: self.n_timesteps,
                omega_phys: self.omega_phys,
                dt: self.dt,
                g: self.g,
                drag: self.drag,
                pixel_sigma: self.pixel_sigma,
                physics_sigma: self.physics_sigma,
            };
            let mut prob_minus = TrajectoryProblem {
                params: params_minus.clone(),
                P_list: self.P_list.clone(),
                pixels: self.pixels.clone(),
                n_timesteps: self.n_timesteps,
                omega_phys: self.omega_phys,
                dt: self.dt,
                g: self.g,
                drag: self.drag,
                pixel_sigma: self.pixel_sigma,
                physics_sigma: self.physics_sigma,
            };
            if let (Some(r_plus), Some(r_minus)) = (prob_plus.residuals(), prob_minus.residuals()) {
                for i in 0..m {
                    J[(i, j)] = (r_plus[i] - r_minus[i]) / (2.0 * epsilon)
                }
            }
            params_plus[j] = xj;
            params_minus[j] = xj;
        }
        J
    }

    fn covariance_from_jacobian(&self, jacobian: &DMatrix<f64>) -> Option<DMatrix<f64>> {
        let r = self.residuals()?;
        let m = r.len();
        let n = self.params.len();
        let jtj = jacobian.transpose() * jacobian;
        let jtj_inv = jtj.try_inverse()?;
        let dof = (m - n).max(1) as f64;
        let sigma2 = r.norm_squared() / dof;
        Some(jtj_inv * sigma2)
    }
}

impl LeastSquaresProblem<f64, Dyn, Dyn> for TrajectoryProblem {
    type ParameterStorage = VecStorage<f64, Dyn, U1>;
    type ResidualStorage = VecStorage<f64, Dyn, U1>;
    type JacobianStorage = VecStorage<f64, Dyn, Dyn>;

    fn set_params(&mut self, x: &DVector<f64>) {
        self.params.copy_from(x);
    }

    fn params(&self) -> DVector<f64> {
        self.params.clone()
    }

    fn residuals(&self) -> Option<DVector<f64>> {
        let n_cams = self.P_list.len();
        let mut residuals = Vec::with_capacity(self.n_timesteps * n_cams * 2 + self.n_timesteps.saturating_sub(2) * 3);
        for t in 0..self.n_timesteps {
            let X_t = Vector3::new(self.params[t * 3], self.params[t * 3 + 1], self.params[t * 3 + 2]);
            for c in 0..n_cams {
                let pred = project_point(self.P_list[c], X_t);
                let pix = self.pixels.get(c).and_then(|row| row.get(t)).copied().unwrap_or(Vector2::zeros());
                let r = (pred - pix) / self.pixel_sigma;
                residuals.push(r[0]);
                residuals.push(r[1]);
            }
        }
        for t in 1..self.n_timesteps.saturating_sub(1) {
            let X_prev = Vector3::new(self.params[(t - 1) * 3], self.params[(t - 1) * 3 + 1], self.params[(t - 1) * 3 + 2]);
            let X_curr = Vector3::new(self.params[t * 3], self.params[t * 3 + 1], self.params[t * 3 + 2]);
            let X_next = Vector3::new(self.params[(t + 1) * 3], self.params[(t + 1) * 3 + 1], self.params[(t + 1) * 3 + 2]);
            let phys_res = (X_next - 2.0 * X_curr + X_prev - self.g * self.dt * self.dt - Vector3::new(self.drag, self.drag, self.drag) * self.dt * self.dt) / self.physics_sigma;
            residuals.push(self.omega_phys * phys_res[0]);
            residuals.push(self.omega_phys * phys_res[1]);
            residuals.push(self.omega_phys * phys_res[2]);
        }
        Some(DVector::from_vec(residuals))
    }

    fn jacobian(&self) -> Option<DMatrix<f64>> {
        None
    }
}

async fn optimize_trajectory(P_list: &[Matrix3x4<f64>], pixels: &[Vec<Vector2<f64>>], dt: f64, g: Option<Vector3<f64>>, drag: f64, pixel_sigma: f64, physics_sigma: f64, omega_phys: f64) -> (Vec<Vector3<f64>>, DMatrix<f64>) {
    let g = g.unwrap_or_else(|| Vector3::new(0.0, 0.0, -9.81));
    let n_timesteps = pixels[0].len();
    let mut X_init = Vec::with_capacity(n_timesteps);
    for t in 0..n_timesteps {
        let mut pixel_t = Vec::new();
        for p in pixels {
            if let Some(v) = p.get(t) {
                pixel_t.push(*v);
            }
        }
        X_init.push(dlt_triangulation(P_list, &pixel_t));
    }
    let params_init: Vec<f64> = X_init.iter().flat_map(|v| vec![v[0], v[1], v[2]]).collect();
    let mut problem = TrajectoryProblem {
        params: DVector::from_vec(params_init),
        P_list: P_list.to_vec(),
        pixels: pixels.to_vec(),
        n_timesteps,
        omega_phys,
        dt,
        g,
        drag,
        pixel_sigma,
        physics_sigma,
    };
    let (problem, _report) = LevenbergMarquardt::default().minimize(problem);
    let X_opt: Vec<Vector3<f64>> = (0..n_timesteps)
        .map(|t| Vector3::new(problem.params[t * 3], problem.params[t * 3 + 1], problem.params[t * 3 + 2]))
        .collect();
    let jacobian = problem.numerical_jacobian(1e-7);
    let cov = problem.covariance_from_jacobian(&jacobian).unwrap_or_else(|| DMatrix::zeros(problem.params.len(), problem.params.len()));
    (X_opt, cov)
}

//==============================================
// Example usage
//==============================================

fn _camera_look_at(cam_position: Vector3<f64>, target: Vector3<f64>, up: Vector3<f64>) -> (Matrix3<f64>, Vector3<f64>) {
    let z = (target - cam_position) / (target - cam_position).norm();
    let x = up.cross(&z).normalize();
    let y = z.cross(&x).normalize();
    let R = Matrix3::new(x[0], y[0], z[0], x[1], y[1], z[1], x[2], y[2], z[2]);
    let t = -R * cam_position;
    (R, t)
}

pub async fn run() {
    print("Starting Physics-Informed Triangulation (no GCPs, fixed cameras)")
    rand::rng().set_seed(43);
    let n_cameras = 3;  
    let n_timesteps = 25;
    let dt = 0.04;
    let g = Vector3::new(0.0, 0.0, -9.81);
    let drag_coef = 0.2;

    let K_list = [Matrix3::new(800.0, 0.0, 0.0, 0.0, 800.0, 0.0, 0.0, 0.0, 1.0) for _ in 0..n_cameras];
    let center = Vector3::new(0.0, 0.0, 0.0);
    let radius = 4.0;
    let height = 2.0;
    let angles = [0.0, 2.0 * std::f64::consts::PI / 3.0, 4.0 * std::f64::consts::PI / 3.0];
    let cam_positions = Matrix3::new(radius * angles[0].cos(), radius * angles[0].sin(), height, radius * angles[1].cos(), radius * angles[1].sin(), height, radius * angles[2].cos(), radius * angles[2].sin(), height);
    let mut R_list = *[Matrix3<f64>];
    let mut t_list = *[Vector3<f64>];
    for i in 0..n_cameras {
        let (R, t) = _camera_look_at(cam_positions[i], center);
        R_list.push(*R);
        t_list.push(*t);
    }

    let P_list = [K_list[i] @ Matrix3x4::new(R_list[i], t_list[i].reshape(3, 1)) for i in 0..n_cameras]; //may be problem with the reshape

    let X0 = Vector3::new(0.0, 0.0, 1.5);
    let V0 = Vector3::new(1.0, 1.0, 2.5);
    let mut traj_true = Vec::new();
    let mut x = X0.clone();
    let mut v = V0.clone();
    for t in 0..n_timesteps {
        traj_true.push(x.clone());
        acc = g - drag_coef * v
        v = v + acc * dt;
        x = x + v * dt;
        if x[2] < 0.2 {
            break;
        }
    }
    let n_timesteps = traj_true.len();
    let traj_true = Array2::from(traj_true);

    let mut pixels = Vec::new();
    for i in 0..n_cameras {
        let mut pixel_t = Vec::new();
        for t in 0..n_timesteps {
            let pred = project_point(P_list[i], traj_true[t]);
            pixel_t.push(*pred);
        }
        pixels.push(pixel_t);
    }

    let time_start = Instant::now();
    let (X_opt, _cov) = optimize_trajectory(&P_list, &pixels, dt, g, drag, pixel_sigma, physics_sigma, omega_phys).await;

    let time_end = Instant::now();
    println!("Time taken: {:.4f} s", time_end.duration_since(time_start).as_secs_f64());
    println!("This is X: \n{:?}", X_opt);
    println!("This is covariance: \n{:?}", _cov);


    // Plot: true vs optimized trajectory


}
