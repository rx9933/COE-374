

pub async fn run() {
    let P_list = vec![
        Matrix3x4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        Matrix3x4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
        Matrix3x4::new(1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0),
    ];
    let pixels = vec![
        Vector2::new(1.0, 1.0),
        Vector2::new(2.0, 3.0),
        Vector2::new(6.0, 4.0),
    ];
    let X = dlt_triangulation(&P_list, &pixels).await;
    println!("This is X: \n{:?}", X);

    let P_list = vec![
        Matrix3x4::new(
            -1_245.78145, 761.33342, 0.0, 24_000.00183,
            -686.836277, -215.999835, 1_100.0, 15_800.00137,
            -0.953939273, -0.299999771, 0.0, 25.000019,
        ),
        Matrix3x4::new(
            -391.564502, 5_154.90807, 0.0, 48_000.00366,
            -1_030.25442, 323.999753, 4_800.0, 17_400.00206,
            -0.953939273, 0.299999771, 0.0, 25.000019,
        ),
    ];
    let pixels = vec![
        Vector2::new(0.0, 0.0),
        Vector2::new(0.0, 0.0),
    ];
    let X = dlt_triangulation(&P_list, &pixels).await;
    println!("This is X: \n{:?}", X);
}


pub async fn run() {
    let P_list = vec![
        Matrix3x4::new(
            -1_245.78145, 761.33342, 0.0, 24_000.00183,
            -686.836277, -215.999835, 1_100.0, 15_800.00137,
            -0.953939273, -0.299999771, 0.0, 25.000019,
        ),
        Matrix3x4::new(
            -391.564502, 5_154.90807, 0.0, 48_000.00366,
            -1_030.25442, 323.999753, 4_800.0, 17_400.00206,
            -0.953939273, 0.299999771, 0.0, 25.000019,
        ),
    ];
    let pixels: Vec<Vec<Vector2<f64>>> = vec![
        vec![Vector2::new(0.0, 0.0)],
        vec![Vector2::new(0.0, 0.0)],
    ];
    let dt = 1.0;
    let g = Some(Vector3::new(0.0, 0.0, -9.81));
    let drag = 0.0;
    let pixel_sigma = 1.0;
    let physics_sigma = 0.01;
    let omega_phys = 1.0;
    let (X_opt, _cov) = optimize_trajectory(&P_list, &pixels, dt, g, drag, pixel_sigma, physics_sigma, omega_phys).await;
    println!("This is X: \n{:?}", X_opt);
    println!("This is covariance: \n{:?}", _cov);
}
