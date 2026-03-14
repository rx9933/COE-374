mod math_triangulation;

#[tokio::main]
async fn main() {
    math_triangulation::run().await;
}
