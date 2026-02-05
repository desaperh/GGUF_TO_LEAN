/// GGUF file reader with complete dequantization support including K-quants
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use byteorder::{LittleEndian, ReadBytesExt};
use half::f16;

use crate::types::*;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub struct GGUFReader {
    file: File,
    version: u32,
    tensor_count: u64,
    metadata_count: u64,
    alignment: u32,
    data_offset: u64,
    metadata: std::collections::HashMap<String, MetadataValue>,
    tensor_infos: Vec<TensorInfo>,
}

impl GGUFReader {
    /// Open a GGUF file
    pub fn open<P: AsRef<Path>>(path: P) -> Result<Self> {
        let file = File::open(path)?;
        Ok(Self {
            file,
            version: 0,
            tensor_count: 0,
            metadata_count: 0,
            alignment: 32,
            data_offset: 0,
            metadata: std::collections::HashMap::new(),
            tensor_infos: Vec::new(),
        })
    }
    
    /// Parse the GGUF file header and metadata
    pub fn parse(&mut self) -> Result<()> {
        // Read magic number
        let magic = self.file.read_u32::<LittleEndian>()?;
        if magic != 0x46554747 {  // "GGUF"
            return Err("Not a valid GGUF file".into());
        }
        
        // Read version
        self.version = self.file.read_u32::<LittleEndian>()?;
        
        // Read counts
        self.tensor_count = self.file.read_u64::<LittleEndian>()?;
        self.metadata_count = self.file.read_u64::<LittleEndian>()?;
        
        // Read metadata
        for _ in 0..self.metadata_count {
            let key = self.read_string()?;
            let value_type = GGUFValueType::try_from(
                self.file.read_u32::<LittleEndian>()?
            )?;
            let value = self.read_value(value_type)?;
            self.metadata.insert(key, value);
        }
        
        // Extract alignment from metadata
        if let Some(MetadataValue::UInt32(align)) = self.metadata.get("general.alignment") {
            self.alignment = *align;
        }
        
        // Read tensor infos
        self.tensor_infos.reserve(self.tensor_count as usize);
        for _ in 0..self.tensor_count {
            let name = self.read_string()?;
            let n_dims = self.file.read_u32::<LittleEndian>()?;
            
            let mut dimensions = Vec::with_capacity(n_dims as usize);
            for _ in 0..n_dims {
                dimensions.push(self.file.read_u64::<LittleEndian>()?);
            }
            
            let tensor_type = GGMLType::try_from(
                self.file.read_u32::<LittleEndian>()?
            )?;
            let offset = self.file.read_u64::<LittleEndian>()?;
            
            self.tensor_infos.push(TensorInfo {
                name,
                dimensions,
                tensor_type,
                offset,
            });
        }
        
        // Calculate data offset (aligned)
        let current_pos = self.file.stream_position()?;
        self.data_offset = ((current_pos + self.alignment as u64 - 1) / self.alignment as u64) 
                          * self.alignment as u64;
        
        Ok(())
    }
    
    /// Get metadata
    pub fn metadata(&self) -> &std::collections::HashMap<String, MetadataValue> {
        &self.metadata
    }
    
    /// Get tensor infos
    pub fn tensor_infos(&self) -> &[TensorInfo] {
        &self.tensor_infos
    }
    
    /// Load and dequantize a tensor
    pub fn load_tensor(&mut self, info: &TensorInfo) -> Result<Tensor> {
        // Seek to tensor data
        self.file.seek(SeekFrom::Start(self.data_offset + info.offset))?;
        
        // Read and dequantize based on type
        let data = match info.tensor_type {
            GGMLType::F32 => self.read_f32(info.n_elements())?,
            GGMLType::F16 => self.read_f16(info.n_elements())?,
            GGMLType::Q4_0 => self.dequantize_q4_0(info.n_elements())?,
            GGMLType::Q4_1 => self.dequantize_q4_1(info.n_elements())?,
            GGMLType::Q5_0 => self.dequantize_q5_0(info.n_elements())?,
            GGMLType::Q5_1 => self.dequantize_q5_1(info.n_elements())?,
            GGMLType::Q8_0 => self.dequantize_q8_0(info.n_elements())?,
            GGMLType::Q8_1 => self.dequantize_q8_1(info.n_elements())?,
            // K-quants (256 elements per block)
            GGMLType::Q2_K => self.dequantize_q2_k(info.n_elements())?,
            GGMLType::Q3_K => self.dequantize_q3_k(info.n_elements())?,
            GGMLType::Q4_K => self.dequantize_q4_k(info.n_elements())?,
            GGMLType::Q5_K => self.dequantize_q5_k(info.n_elements())?,
            GGMLType::Q6_K => self.dequantize_q6_k(info.n_elements())?,
            GGMLType::Q8_K => self.dequantize_q8_k(info.n_elements())?,
            _ => return Err(format!("Unsupported tensor type: {:?}", info.tensor_type).into()),
        };
        
        Ok(Tensor {
            name: info.name.clone(),
            dimensions: info.dimensions.clone(),
            data,
        })
    }
    
    // ========================================================================
    // Private helper methods
    // ========================================================================
    
    fn read_string(&mut self) -> Result<String> {
        let length = self.file.read_u64::<LittleEndian>()? as usize;
        let mut bytes = vec![0u8; length];
        self.file.read_exact(&mut bytes)?;
        Ok(String::from_utf8(bytes)?)
    }
    
    fn read_value(&mut self, value_type: GGUFValueType) -> Result<MetadataValue> {
        Ok(match value_type {
            GGUFValueType::UInt8 => MetadataValue::UInt8(self.file.read_u8()?),
            GGUFValueType::Int8 => MetadataValue::Int8(self.file.read_i8()?),
            GGUFValueType::UInt16 => MetadataValue::UInt16(self.file.read_u16::<LittleEndian>()?),
            GGUFValueType::Int16 => MetadataValue::Int16(self.file.read_i16::<LittleEndian>()?),
            GGUFValueType::UInt32 => MetadataValue::UInt32(self.file.read_u32::<LittleEndian>()?),
            GGUFValueType::Int32 => MetadataValue::Int32(self.file.read_i32::<LittleEndian>()?),
            GGUFValueType::Float32 => MetadataValue::Float32(self.file.read_f32::<LittleEndian>()?),
            GGUFValueType::Bool => MetadataValue::Bool(self.file.read_u8()? != 0),
            GGUFValueType::String => MetadataValue::String(self.read_string()?),
            GGUFValueType::UInt64 => MetadataValue::UInt64(self.file.read_u64::<LittleEndian>()?),
            GGUFValueType::Int64 => MetadataValue::Int64(self.file.read_i64::<LittleEndian>()?),
            GGUFValueType::Float64 => MetadataValue::Float64(self.file.read_f64::<LittleEndian>()?),
            GGUFValueType::Array => {
                let array_type = GGUFValueType::try_from(self.file.read_u32::<LittleEndian>()?)?;
                let array_len = self.file.read_u64::<LittleEndian>()?;
                let mut strings = Vec::new();
                
                for _ in 0..array_len {
                    if let MetadataValue::String(s) = self.read_value(array_type)? {
                        strings.push(s);
                    }
                }
                MetadataValue::StringArray(strings)
            }
        })
    }
    
    // ========================================================================
    // Basic dequantization methods (32 elements per block)
    // ========================================================================
    
    fn read_f32(&mut self, n_elements: usize) -> Result<Vec<f32>> {
        let mut data = Vec::with_capacity(n_elements);
        for _ in 0..n_elements {
            data.push(self.file.read_f32::<LittleEndian>()?);
        }
        Ok(data)
    }
    
    fn read_f16(&mut self, n_elements: usize) -> Result<Vec<f32>> {
        let mut data = Vec::with_capacity(n_elements);
        for _ in 0..n_elements {
            let f16_val = f16::from_bits(self.file.read_u16::<LittleEndian>()?);
            data.push(f16_val.to_f32());
        }
        Ok(data)
    }
    
    fn dequantize_q4_0(&mut self, n_elements: usize) -> Result<Vec<f32>> {
        let n_blocks = n_elements / 32;
        let mut result = Vec::with_capacity(n_elements);
        
        for _ in 0..n_blocks {
            let delta = self.file.read_f32::<LittleEndian>()?;
            let mut qs = [0u8; 16];
            self.file.read_exact(&mut qs)?;
            
            for byte in qs.iter() {
                let val0 = (byte & 0x0F) as i8 - 8;
                result.push(val0 as f32 * delta);
                
                let val1 = (byte >> 4) as i8 - 8;
                result.push(val1 as f32 * delta);
            }
        }
        
        Ok(result)
    }
    
    fn dequantize_q4_1(&mut self, n_elements: usize) -> Result<Vec<f32>> {
        let n_blocks = n_elements / 32;
        let mut result = Vec::with_capacity(n_elements);
        
        for _ in 0..n_blocks {
            let delta = self.file.read_f32::<LittleEndian>()?;
            let min = self.file.read_f32::<LittleEndian>()?;
            let mut qs = [0u8; 16];
            self.file.read_exact(&mut qs)?;
            
            for byte in qs.iter() {
                let val0 = (byte & 0x0F) as f32;
                result.push(val0 * delta + min);
                
                let val1 = (byte >> 4) as f32;
                result.push(val1 * delta + min);
            }
        }
        
        Ok(result)
    }
    
    fn dequantize_q5_0(&mut self, n_elements: usize) -> Result<Vec<f32>> {
        let n_blocks = n_elements / 32;
        let mut result = Vec::with_capacity(n_elements);
        
        for _ in 0..n_blocks {
            let delta = self.file.read_f32::<LittleEndian>()?;
            let qh = self.file.read_u32::<LittleEndian>()?;
            let mut qs = [0u8; 16];
            self.file.read_exact(&mut qs)?;
            
            for i in 0..16 {
                let byte = qs[i];
                
                // Extract 5-bit values - FIX: cast to u8 before OR operation
                let high_bit_0 = ((qh >> (i * 2)) & 1) as u8;
                let val0 = ((byte & 0x0F) | (high_bit_0 << 4)) as i8 - 16;
                result.push(val0 as f32 * delta);
                
                let high_bit_1 = ((qh >> (i * 2 + 1)) & 1) as u8;
                let val1 = ((byte >> 4) | (high_bit_1 << 4)) as i8 - 16;
                result.push(val1 as f32 * delta);
            }
        }
        
        Ok(result)
    }
    
    fn dequantize_q5_1(&mut self, n_elements: usize) -> Result<Vec<f32>> {
        let n_blocks = n_elements / 32;
        let mut result = Vec::with_capacity(n_elements);
        
        for _ in 0..n_blocks {
            let delta = self.file.read_f32::<LittleEndian>()?;
            let min = self.file.read_f32::<LittleEndian>()?;
            let qh = self.file.read_u32::<LittleEndian>()?;
            let mut qs = [0u8; 16];
            self.file.read_exact(&mut qs)?;
            
            for i in 0..16 {
                let byte = qs[i];
                
                // FIX: cast to u8 before OR operation
                let high_bit_0 = ((qh >> (i * 2)) & 1) as u8;
                let val0 = ((byte & 0x0F) | (high_bit_0 << 4)) as f32;
                result.push(val0 * delta + min);
                
                let high_bit_1 = ((qh >> (i * 2 + 1)) & 1) as u8;
                let val1 = ((byte >> 4) | (high_bit_1 << 4)) as f32;
                result.push(val1 * delta + min);
            }
        }
        
        Ok(result)
    }
    
    fn dequantize_q8_0(&mut self, n_elements: usize) -> Result<Vec<f32>> {
        let n_blocks = n_elements / 32;
        let mut result = Vec::with_capacity(n_elements);
        
        for _ in 0..n_blocks {
            let delta = self.file.read_f32::<LittleEndian>()?;
            let mut qs = [0i8; 32];
            for q in qs.iter_mut() {
                *q = self.file.read_i8()?;
            }
            
            for q in qs.iter() {
                result.push(*q as f32 * delta);
            }
        }
        
        Ok(result)
    }
    
    fn dequantize_q8_1(&mut self, n_elements: usize) -> Result<Vec<f32>> {
        let n_blocks = n_elements / 32;
        let mut result = Vec::with_capacity(n_elements);
        
        for _ in 0..n_blocks {
            let delta = self.file.read_f32::<LittleEndian>()?;
            let min = self.file.read_f32::<LittleEndian>()?;
            let mut qs = [0i8; 32];
            for q in qs.iter_mut() {
                *q = self.file.read_i8()?;
            }
            
            for q in qs.iter() {
                result.push(*q as f32 * delta + min);
            }
        }
        
        Ok(result)
    }
    
    // ========================================================================
    // K-quants dequantization methods (256 elements per block)
    // ========================================================================
    
    fn dequantize_q2_k(&mut self, n_elements: usize) -> Result<Vec<f32>> {
        let n_blocks = n_elements / 256;
        let mut result = Vec::with_capacity(n_elements);
        
        for _ in 0..n_blocks {
            let mut scales_and_mins = [0u8; 16];
            self.file.read_exact(&mut scales_and_mins)?;
            
            let mut qs = [0u8; 64];
            self.file.read_exact(&mut qs)?;
            
            // FIX: Read f16 properly using half crate
            let d = f16::from_bits(self.file.read_u16::<LittleEndian>()?).to_f32();
            let dmin = f16::from_bits(self.file.read_u16::<LittleEndian>()?).to_f32();
            
            for i in 0..64 {
                let byte = qs[i];
                let scale_idx = i / 16;
                let scale = ((scales_and_mins[scale_idx / 2] >> ((scale_idx % 2) * 4)) & 0x0F) as f32;
                let min = ((scales_and_mins[8 + scale_idx / 2] >> ((scale_idx % 2) * 4)) & 0x0F) as f32;
                
                for j in 0..4 {
                    let val = ((byte >> (j * 2)) & 0x03) as f32;
                    result.push(d * scale * val - dmin * min);
                }
            }
        }
        
        Ok(result)
    }
    
    fn dequantize_q3_k(&mut self, n_elements: usize) -> Result<Vec<f32>> {
        let n_blocks = n_elements / 256;
        let mut result = Vec::with_capacity(n_elements);
        
        for _ in 0..n_blocks {
            let mut hmask = [0u8; 32];
            self.file.read_exact(&mut hmask)?;
            
            let mut qs = [0u8; 64];
            self.file.read_exact(&mut qs)?;
            
            let mut scales = [0u8; 12];
            self.file.read_exact(&mut scales)?;
            
            let d = f16::from_bits(self.file.read_u16::<LittleEndian>()?).to_f32();
            
            for i in 0..64 {
                let byte = qs[i];
                let scale_idx = i / 16;
                let scale = scales[scale_idx] as f32;
                
                for j in 0..4 {
                    let val = ((byte >> (j * 2)) & 0x03) as i8;
                    let high_bit = ((hmask[(i * 4 + j) / 8] >> ((i * 4 + j) % 8)) & 1) as i8;
                    let combined = val | (high_bit << 2);
                    result.push(d * scale * (combined - 4) as f32);
                }
            }
        }
        
        Ok(result)
    }
    
    fn dequantize_q4_k(&mut self, n_elements: usize) -> Result<Vec<f32>> {
        let n_blocks = n_elements / 256;
        let mut result = Vec::with_capacity(n_elements);
        
        for _ in 0..n_blocks {
            let d = f16::from_bits(self.file.read_u16::<LittleEndian>()?).to_f32();
            let dmin = f16::from_bits(self.file.read_u16::<LittleEndian>()?).to_f32();
            
            let mut scales = [0u8; 12];
            self.file.read_exact(&mut scales)?;
            
            let mut qs = [0u8; 128];
            self.file.read_exact(&mut qs)?;
            
            for i in 0..128 {
                let byte = qs[i];
                let scale_idx = i / 32;
                let scale = scales[scale_idx] as f32;
                
                let val0 = (byte & 0x0F) as f32;
                result.push(d * scale * val0 - dmin);
                
                let val1 = (byte >> 4) as f32;
                result.push(d * scale * val1 - dmin);
            }
        }
        
        Ok(result)
    }
    
    fn dequantize_q5_k(&mut self, n_elements: usize) -> Result<Vec<f32>> {
        let n_blocks = n_elements / 256;
        let mut result = Vec::with_capacity(n_elements);
        
        for _ in 0..n_blocks {
            let d = f16::from_bits(self.file.read_u16::<LittleEndian>()?).to_f32();
            let dmin = f16::from_bits(self.file.read_u16::<LittleEndian>()?).to_f32();
            
            let mut scales = [0u8; 12];
            self.file.read_exact(&mut scales)?;
            
            let mut qh = [0u8; 32];
            self.file.read_exact(&mut qh)?;
            
            let mut qs = [0u8; 128];
            self.file.read_exact(&mut qs)?;
            
            for i in 0..128 {
                let byte = qs[i];
                let scale_idx = i / 32;
                let scale = scales[scale_idx] as f32;
                
                // FIX: cast high bit to u8 before OR
                let high_bit_0 = ((qh[i / 8] >> (i % 8)) & 1) as u8;
                let val0 = ((byte & 0x0F) | (high_bit_0 << 4)) as f32;
                result.push(d * scale * val0 - dmin);
                
                let high_bit_1 = ((qh[64 + i / 8] >> (i % 8)) & 1) as u8;
                let val1 = ((byte >> 4) | (high_bit_1 << 4)) as f32;
                result.push(d * scale * val1 - dmin);
            }
        }
        
        Ok(result)
    }
    
    fn dequantize_q6_k(&mut self, n_elements: usize) -> Result<Vec<f32>> {
        let n_blocks = n_elements / 256;
        let mut result = Vec::with_capacity(n_elements);
        
        for _ in 0..n_blocks {
            let mut ql = [0u8; 128];
            self.file.read_exact(&mut ql)?;
            
            let mut qh = [0u8; 64];
            self.file.read_exact(&mut qh)?;
            
            let mut scales = [0i8; 16];
            for s in scales.iter_mut() {
                *s = self.file.read_i8()?;
            }
            
            let d = f16::from_bits(self.file.read_u16::<LittleEndian>()?).to_f32();
            
            for i in 0..128 {
                let scale_idx = i / 16;
                let scale = scales[scale_idx] as f32;
                
                // FIX: cast high bits to u8 before OR
                let q_low = (ql[i] & 0x0F) as i8;
                let q_high = ((qh[i / 2] >> ((i % 2) * 4)) & 0x03) as i8;
                let q = q_low | (q_high << 4);
                result.push(d * scale * (q - 32) as f32);
                
                let q_low = (ql[i] >> 4) as i8;
                let q_high = ((qh[i / 2] >> ((i % 2) * 4 + 2)) & 0x03) as i8;
                let q = q_low | (q_high << 4);
                result.push(d * scale * (q - 32) as f32);
            }
        }
        
        Ok(result)
    }
    
    fn dequantize_q8_k(&mut self, n_elements: usize) -> Result<Vec<f32>> {
        let n_blocks = n_elements / 256;
        let mut result = Vec::with_capacity(n_elements);
        
        for _ in 0..n_blocks {
            let d = self.file.read_f32::<LittleEndian>()?;
            
            let mut qs = [0i8; 256];
            for q in qs.iter_mut() {
                *q = self.file.read_i8()?;
            }
            
            let mut _bsums = [0i16; 16];
            for bsum in _bsums.iter_mut() {
                *bsum = self.file.read_i16::<LittleEndian>()?;
            }
            
            for q in qs.iter() {
                result.push(d * (*q as f32));
            }
        }
        
        Ok(result)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_q4_0_dequantization() {
        let delta = 0.1f32;
        let byte = 0b10110100u8;
        
        let val0 = ((byte & 0x0F) as i8 - 8) as f32 * delta;
        let val1 = ((byte >> 4) as i8 - 8) as f32 * delta;
        
        assert_eq!(val0, -0.4);
        assert_eq!(val1, 0.3);
    }
    
    #[test]
    fn test_q6_k_bit_extraction() {
        let ql = 0b00001101u8;
        let qh = 0b00000010u8;
        
        let q_low = (ql & 0x0F) as i8;
        let q_high = (qh & 0x03) as i8;
        let q = q_low | (q_high << 4);
        
        assert_eq!(q, 13 | (2 << 4));
    }
}
