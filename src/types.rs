/// Core types for GGUF format and model architecture
use std::collections::HashMap;

// ============================================================================
// GGUF Format Types
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GGUFValueType {
    UInt8 = 0,
    Int8 = 1,
    UInt16 = 2,
    Int16 = 3,
    UInt32 = 4,
    Int32 = 5,
    Float32 = 6,
    Bool = 7,
    String = 8,
    Array = 9,
    UInt64 = 10,
    Int64 = 11,
    Float64 = 12,
}

impl TryFrom<u32> for GGUFValueType {
    type Error = String;
    
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(GGUFValueType::UInt8),
            1 => Ok(GGUFValueType::Int8),
            2 => Ok(GGUFValueType::UInt16),
            3 => Ok(GGUFValueType::Int16),
            4 => Ok(GGUFValueType::UInt32),
            5 => Ok(GGUFValueType::Int32),
            6 => Ok(GGUFValueType::Float32),
            7 => Ok(GGUFValueType::Bool),
            8 => Ok(GGUFValueType::String),
            9 => Ok(GGUFValueType::Array),
            10 => Ok(GGUFValueType::UInt64),
            11 => Ok(GGUFValueType::Int64),
            12 => Ok(GGUFValueType::Float64),
            _ => Err(format!("Unknown GGUF value type: {}", value)),
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u32)]
pub enum GGMLType {
    F32 = 0,
    F16 = 1,
    Q4_0 = 2,
    Q4_1 = 3,
    Q5_0 = 6,
    Q5_1 = 7,
    Q8_0 = 8,
    Q8_1 = 9,
    Q2_K = 10,
    Q3_K = 11,
    Q4_K = 12,
    Q5_K = 13,
    Q6_K = 14,
    Q8_K = 15,
    I8 = 16,
    I16 = 17,
    I32 = 18,
}

impl TryFrom<u32> for GGMLType {
    type Error = String;
    
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(GGMLType::F32),
            1 => Ok(GGMLType::F16),
            2 => Ok(GGMLType::Q4_0),
            3 => Ok(GGMLType::Q4_1),
            6 => Ok(GGMLType::Q5_0),
            7 => Ok(GGMLType::Q5_1),
            8 => Ok(GGMLType::Q8_0),
            9 => Ok(GGMLType::Q8_1),
            10 => Ok(GGMLType::Q2_K),
            11 => Ok(GGMLType::Q3_K),
            12 => Ok(GGMLType::Q4_K),
            13 => Ok(GGMLType::Q5_K),
            14 => Ok(GGMLType::Q6_K),
            15 => Ok(GGMLType::Q8_K),
            16 => Ok(GGMLType::I8),
            17 => Ok(GGMLType::I16),
            18 => Ok(GGMLType::I32),
            _ => Err(format!("Unknown GGML type: {}", value)),
        }
    }
}

impl GGMLType {
    /// Get the block size for quantized types
    pub fn block_size(&self) -> usize {
        match self {
            GGMLType::Q4_0 | GGMLType::Q4_1 | 
            GGMLType::Q5_0 | GGMLType::Q5_1 |
            GGMLType::Q8_0 | GGMLType::Q8_1 => 32,
            GGMLType::Q2_K | GGMLType::Q3_K | 
            GGMLType::Q4_K | GGMLType::Q5_K | 
            GGMLType::Q6_K | GGMLType::Q8_K => 256,
            _ => 1,
        }
    }
    
    /// Get bytes per block for storage
    pub fn bytes_per_block(&self) -> usize {
        match self {
            GGMLType::F32 => 4,
            GGMLType::F16 => 2,
            GGMLType::Q4_0 => 18,  // 4 bytes delta + 16 bytes data
            GGMLType::Q4_1 => 22,  // 8 bytes (delta+min) + 16 bytes data
            GGMLType::Q8_0 => 36,  // 4 bytes delta + 32 bytes data
            GGMLType::Q8_1 => 40,  // 8 bytes (delta+min) + 32 bytes data
            GGMLType::I8 => 1,
            GGMLType::I16 => 2,
            GGMLType::I32 => 4,
            _ => 4,
        }
    }
}

// ============================================================================
// Quantization Block Structures
// ============================================================================

#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ4_0 {
    pub delta: f32,
    pub qs: [u8; 16],  // 32 x 4-bit values packed
}

#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ4_1 {
    pub delta: f32,
    pub min: f32,
    pub qs: [u8; 16],
}

#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ8_0 {
    pub delta: f32,
    pub qs: [i8; 32],
}

#[repr(C, packed)]
#[derive(Debug, Clone, Copy)]
pub struct BlockQ8_1 {
    pub delta: f32,
    pub min: f32,
    pub qs: [i8; 32],
}

// ============================================================================
// Metadata Value
// ============================================================================

#[derive(Debug, Clone)]
pub enum MetadataValue {
    UInt8(u8),
    Int8(i8),
    UInt16(u16),
    Int16(i16),
    UInt32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    UInt64(u64),
    Int64(i64),
    Float64(f64),
    StringArray(Vec<String>),
}

impl MetadataValue {
    pub fn as_u32(&self) -> Option<u32> {
        match self {
            MetadataValue::UInt32(v) => Some(*v),
            MetadataValue::Int32(v) if *v >= 0 => Some(*v as u32),
            _ => None,
        }
    }
    
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            MetadataValue::UInt64(v) => Some(*v),
            MetadataValue::UInt32(v) => Some(*v as u64),
            _ => None,
        }
    }
    
    pub fn as_string(&self) -> Option<&str> {
        match self {
            MetadataValue::String(s) => Some(s.as_str()),
            _ => None,
        }
    }
    
    pub fn as_f32(&self) -> Option<f32> {
        match self {
            MetadataValue::Float32(v) => Some(*v),
            _ => None,
        }
    }
}

// ============================================================================
// Tensor Information
// ============================================================================

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub tensor_type: GGMLType,
    pub offset: u64,
}

impl TensorInfo {
    pub fn n_elements(&self) -> usize {
        self.dimensions.iter().product::<u64>() as usize
    }
    
    pub fn data_size(&self) -> usize {
        let n = self.n_elements();
        let block_size = self.tensor_type.block_size();
        let bytes_per_block = self.tensor_type.bytes_per_block();
        
        if block_size > 1 {
            (n / block_size) * bytes_per_block
        } else {
            n * bytes_per_block
        }
    }
}

// ============================================================================
// Tensor (with dequantized data)
// ============================================================================

#[derive(Debug, Clone)]
pub struct Tensor {
    pub name: String,
    pub dimensions: Vec<u64>,
    pub data: Vec<f32>,
}

impl Tensor {
    pub fn n_elements(&self) -> usize {
        self.data.len()
    }
    
    /// Access element in 2D tensor (matrix)
    pub fn at(&self, row: usize, col: usize) -> Option<f32> {
        if self.dimensions.len() == 2 {
            let cols = self.dimensions[1] as usize;
            self.data.get(row * cols + col).copied()
        } else {
            None
        }
    }
    
    /// Check if tensor is sane (no NaN or Inf)
    pub fn is_sane(&self) -> bool {
        self.data.iter().all(|&x| x.is_finite())
    }
}

// ============================================================================
// Layer Components
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ComponentRole {
    Query,
    Key,
    Value,
    Output,
    MlpGate,
    MlpUp,
    MlpDown,
    AttnNorm,
    FfnNorm,
    Unknown,
}

#[derive(Debug, Clone)]
pub struct LayerComponent {
    pub name: String,
    pub tensor_name: String,
    pub dimensions: Vec<u64>,
    pub role: ComponentRole,
}

// ============================================================================
// Layer Structure
// ============================================================================

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerType {
    TransformerBlock,
    Embedding,
    Output,
}

#[derive(Debug, Clone)]
pub struct Layer {
    pub index: usize,
    pub layer_type: LayerType,
    pub components: HashMap<String, LayerComponent>,
}

impl Layer {
    pub fn new(index: usize) -> Self {
        Self {
            index,
            layer_type: LayerType::TransformerBlock,
            components: HashMap::new(),
        }
    }
    
    pub fn has_component(&self, name: &str) -> bool {
        self.components.contains_key(name)
    }
    
    pub fn has_attention(&self) -> bool {
        self.has_component("q_proj") && 
        self.has_component("k_proj") && 
        self.has_component("v_proj")
    }
    
    pub fn has_mlp(&self) -> bool {
        self.has_component("gate_proj") || 
        self.has_component("up_proj") || 
        self.has_component("down_proj")
    }
}

// ============================================================================
// Model Configuration
// ============================================================================

#[derive(Debug, Clone)]
pub struct ModelConfig {
    pub n_layers: u32,
    pub n_heads: u32,
    pub n_kv_heads: u32,
    pub hidden_size: u32,
    pub vocab_size: u32,
    pub context_length: u32,
    pub intermediate_size: u32,
    pub rope_theta: f32,
    pub architecture: String,
}

impl Default for ModelConfig {
    fn default() -> Self {
        Self {
            n_layers: 0,
            n_heads: 0,
            n_kv_heads: 0,
            hidden_size: 0,
            vocab_size: 0,
            context_length: 0,
            intermediate_size: 0,
            rope_theta: 10000.0,
            architecture: String::from("unknown"),
        }
    }
}

// ============================================================================
// Model Architecture
// ============================================================================

#[derive(Debug, Clone)]
pub struct ModelArchitecture {
    pub config: ModelConfig,
    pub embedding: Option<TensorInfo>,
    pub layers: Vec<Layer>,
    pub output: Option<TensorInfo>,
}

impl ModelArchitecture {
    pub fn new() -> Self {
        Self {
            config: ModelConfig::default(),
            embedding: None,
            layers: Vec::new(),
            output: None,
        }
    }
}

impl Default for ModelArchitecture {
    fn default() -> Self {
        Self::new()
    }
}
