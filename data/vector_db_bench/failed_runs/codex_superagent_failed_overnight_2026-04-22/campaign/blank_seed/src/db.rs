use crate::api::*;

pub struct VectorDB {
    // 模型自行定义内部数据结构
}

impl VectorDB {
    pub fn new() -> Self {
        todo!("模型实现")
    }

    pub fn insert(&self, id: u64, vector: Vec<f32>) {
        todo!("模型实现")
    }

    pub fn bulk_insert(&self, vectors: Vec<(u64, Vec<f32>)>) -> usize {
        todo!("模型实现")
    }

    pub fn search(&self, vector: &[f32], top_k: u32) -> Vec<SearchResult> {
        todo!("模型实现")
    }
}
