use std::sync::Arc;

mod api;
mod db;
mod distance;
mod server;

use db::VectorDB;

fn main() {
    let db = Arc::new(VectorDB::new());
    let port = std::env::var("PORT").unwrap_or_else(|_| "8080".to_string());
    let addr = format!("0.0.0.0:{port}");
    server::run(db, &addr).unwrap();
}
