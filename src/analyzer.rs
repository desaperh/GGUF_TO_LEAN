/// Architecture analyzer - reconstructs model structure from tensor names
use crate::types::*;
use std::collections::HashMap;

pub struct ArchitectureAnalyzer<'a> {
    tensor_infos: &'a [TensorInfo],
    metadata: &'a HashMap<String, MetadataValue>,
}

impl<'a> ArchitectureAnalyzer<'a> {
    pub fn new(
        tensor_infos: &'a [TensorInfo],
        metadata: &'a HashMap<String, MetadataValue>,
    ) -> Self {
        Self {
            tensor_infos,
            metadata,
        }
    }
    
    /// Analyze the model and reconstruct its architecture
    pub fn analyze(&self) -> ModelArchitecture {
        let mut arch = ModelArchitecture::new();
        
        // Extract configuration from metadata
        self.extract_config(&mut arch.config);
        
        // Categorize tensors into layers
        self.categorize_tensors(&mut arch);
        
        arch
    }
    
    fn extract_config(&self, config: &mut ModelConfig) {
        // Get architecture name first
        let arch_name = self.metadata.get("general.architecture")
            .and_then(|v| v.as_string())
            .unwrap_or("unknown");
        config.architecture = arch_name.to_string();
        
        println!("🔍 Detected architecture: {}", arch_name);
        
        // Try multiple naming schemes based on architecture
        let prefixes = match arch_name {
            "llama" => vec!["llama"],
            "mistral" | "mistral3" => vec!["mistral", "llama"], // Mistral uses llama format
            "gpt2" => vec!["gpt2"],
            "phi" | "phi2" | "phi3" => vec!["phi", "llama"],
            _ => vec!["llama", "mistral", "gpt2"], // Try all
        };
        
        println!("🔍 Trying metadata prefixes: {:?}", prefixes);
        
        // Try each prefix until we find values
        for prefix in &prefixes {
            // Core parameters
            if config.n_layers == 0 {
                config.n_layers = self.get_metadata_u32(&format!("{}.block_count", prefix), 0);
            }
            
            if config.n_heads == 0 {
                config.n_heads = self.get_metadata_u32(&format!("{}.attention.head_count", prefix), 0);
            }
            
            if config.n_kv_heads == 0 {
                config.n_kv_heads = self.get_metadata_u32(&format!("{}.attention.head_count_kv", prefix), 0);
            }
            
            if config.hidden_size == 0 {
                config.hidden_size = self.get_metadata_u32(&format!("{}.embedding_length", prefix), 0);
            }
            
            if config.vocab_size == 0 {
                config.vocab_size = self.get_metadata_u32(&format!("{}.vocab_size", prefix), 0);
            }
            
            if config.context_length == 0 {
                config.context_length = self.get_metadata_u32(&format!("{}.context_length", prefix), 0);
            }
            
            if config.intermediate_size == 0 {
                config.intermediate_size = self.get_metadata_u32(&format!("{}.feed_forward_length", prefix), 0);
            }
            
            if config.rope_theta == 10000.0 {
                if let Some(theta) = self.get_metadata_f32(&format!("{}.rope.freq_base", prefix)) {
                    config.rope_theta = theta;
                }
            }
            
            // Break early if we found the main parameters
            if config.n_layers > 0 && config.hidden_size > 0 {
                println!("✅ Found config with prefix: {}", prefix);
                break;
            }
        }
        
        // Fallback: try to infer from tensor names
        if config.n_layers == 0 {
            println!("⚠️  No metadata found, inferring from tensor names...");
            config.n_layers = self.infer_layer_count();
            println!("   Inferred layers: {}", config.n_layers);
        }
        
        if config.hidden_size == 0 {
            config.hidden_size = self.infer_hidden_size();
            println!("   Inferred hidden size: {}", config.hidden_size);
        }
        
        if config.vocab_size == 0 {
            config.vocab_size = self.infer_vocab_size();
            println!("   Inferred vocab size: {}", config.vocab_size);
        }
        
        // Print found config
        println!("\n📊 Extracted configuration:");
        println!("   n_layers: {}", config.n_layers);
        println!("   n_heads: {}", config.n_heads);
        println!("   n_kv_heads: {}", config.n_kv_heads);
        println!("   hidden_size: {}", config.hidden_size);
        println!("   vocab_size: {}", config.vocab_size);
        println!("   context_length: {}", config.context_length);
        println!("   intermediate_size: {}", config.intermediate_size);
    }
    
    fn infer_layer_count(&self) -> u32 {
        let mut max_layer = 0;
        for info in self.tensor_infos {
            if let Some(layer_idx) = self.extract_layer_number(&info.name) {
                max_layer = max_layer.max(layer_idx);
            }
        }
        (max_layer + 1) as u32
    }
    
    fn infer_hidden_size(&self) -> u32 {
        // Look for embedding tensor
        for info in self.tensor_infos {
            if info.name.contains("token_embd") || info.name.contains("embed_tokens") {
                if info.dimensions.len() == 2 {
                    return info.dimensions[1] as u32;
                }
            }
        }
        0
    }
    
    fn infer_vocab_size(&self) -> u32 {
        // Look for embedding tensor
        for info in self.tensor_infos {
            if info.name.contains("token_embd") || info.name.contains("embed_tokens") {
                if info.dimensions.len() == 2 {
                    return info.dimensions[0] as u32;
                }
            }
        }
        0
    }
    
    fn categorize_tensors(&self, arch: &mut ModelArchitecture) {
        for info in self.tensor_infos {
            let name = &info.name;
            
            // Check for embedding
            if name.contains("token_embd") || name.contains("embed_tokens") || name.contains("wte") {
                arch.embedding = Some(info.clone());
                continue;
            }
            
            // Check for output
            if name.contains("output") || name.contains("lm_head") {
                arch.output = Some(info.clone());
                continue;
            }
            
            // Check for layer tensors
            if let Some(layer_idx) = self.extract_layer_number(name) {
                // Ensure we have enough layers
                while arch.layers.len() <= layer_idx {
                    arch.layers.push(Layer::new(arch.layers.len()));
                }
                
                // Determine component
                let role = self.determine_role(name);
                let comp_name = Self::role_to_component_name(role, name);
                
                let component = LayerComponent {
                    name: comp_name.clone(),
                    tensor_name: name.clone(),
                    dimensions: info.dimensions.clone(),
                    role,
                };
                
                arch.layers[layer_idx].components.insert(comp_name, component);
            }
        }
    }
    
    fn extract_layer_number(&self, name: &str) -> Option<usize> {
        // Look for patterns like "blk.N." or "layers.N."
        if let Some(pos) = name.find("blk.") {
            let rest = &name[pos + 4..];
            if let Some(dot_pos) = rest.find('.') {
                return rest[..dot_pos].parse().ok();
            }
        }
        
        if let Some(pos) = name.find("layers.") {
            let rest = &name[pos + 7..];
            if let Some(dot_pos) = rest.find('.') {
                return rest[..dot_pos].parse().ok();
            }
        }
        
        None
    }
    
    fn determine_role(&self, name: &str) -> ComponentRole {
        // Attention components
        if name.contains("attn_q") || name.contains("q_proj") {
            return ComponentRole::Query;
        }
        if name.contains("attn_k") || name.contains("k_proj") {
            return ComponentRole::Key;
        }
        if name.contains("attn_v") || name.contains("v_proj") {
            return ComponentRole::Value;
        }
        if name.contains("attn_output") || name.contains("o_proj") {
            return ComponentRole::Output;
        }
        
        // MLP components
        if name.contains("ffn_gate") || name.contains("gate_proj") {
            return ComponentRole::MlpGate;
        }
        if name.contains("ffn_up") || name.contains("up_proj") {
            return ComponentRole::MlpUp;
        }
        if name.contains("ffn_down") || name.contains("down_proj") {
            return ComponentRole::MlpDown;
        }
        
        // Normalization
        if name.contains("attn_norm") || name.contains("input_layernorm") {
            return ComponentRole::AttnNorm;
        }
        if name.contains("ffn_norm") || name.contains("post_attention_layernorm") {
            return ComponentRole::FfnNorm;
        }
        
        ComponentRole::Unknown
    }
    
    fn role_to_component_name(role: ComponentRole, full_name: &str) -> String {
        match role {
            ComponentRole::Query => "q_proj".to_string(),
            ComponentRole::Key => "k_proj".to_string(),
            ComponentRole::Value => "v_proj".to_string(),
            ComponentRole::Output => "o_proj".to_string(),
            ComponentRole::MlpGate => "gate_proj".to_string(),
            ComponentRole::MlpUp => "up_proj".to_string(),
            ComponentRole::MlpDown => "down_proj".to_string(),
            ComponentRole::AttnNorm => "attn_norm".to_string(),
            ComponentRole::FfnNorm => "ffn_norm".to_string(),
            ComponentRole::Unknown => {
                // Use last part of name
                full_name.split('.').last().unwrap_or("unknown").to_string()
            }
        }
    }
    
    fn get_metadata_u32(&self, key: &str, default: u32) -> u32 {
        self.metadata.get(key).and_then(|v| v.as_u32()).unwrap_or(default)
    }
    
    fn get_metadata_f32(&self, key: &str) -> Option<f32> {
        self.metadata.get(key).and_then(|v| v.as_f32())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_layer_number_extraction() {
        let analyzer = ArchitectureAnalyzer {
            tensor_infos: &[],
            metadata: &HashMap::new(),
        };
        
        assert_eq!(analyzer.extract_layer_number("blk.0.attn_q.weight"), Some(0));
        assert_eq!(analyzer.extract_layer_number("blk.5.ffn_gate.weight"), Some(5));
        assert_eq!(analyzer.extract_layer_number("layers.10.attention.q.weight"), Some(10));
        assert_eq!(analyzer.extract_layer_number("token_embd.weight"), None);
    }
    
    #[test]
    fn test_role_determination() {
        let analyzer = ArchitectureAnalyzer {
            tensor_infos: &[],
            metadata: &HashMap::new(),
        };
        
        assert_eq!(analyzer.determine_role("blk.0.attn_q.weight"), ComponentRole::Query);
        assert_eq!(analyzer.determine_role("blk.0.ffn_gate.weight"), ComponentRole::MlpGate);
        assert_eq!(analyzer.determine_role("blk.0.attn_norm.weight"), ComponentRole::AttnNorm);
    }
}
