/// Example: Visualize GGUF model architecture
use gguf_to_lean::{GGUFReader, ArchitectureAnalyzer};
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 2 {
        eprintln!("Usage: {} <model.gguf>", args[0]);
        return Ok(());
    }
    
    let gguf_path = &args[1];
    
    println!("📂 Reading GGUF file: {}", gguf_path);
    
    // Parse GGUF
    let mut reader = GGUFReader::open(gguf_path)?;
    reader.parse()?;
    
    // Analyze architecture
    let analyzer = ArchitectureAnalyzer::new(
        reader.tensor_infos(),
        reader.metadata(),
    );
    let architecture = analyzer.analyze();
    
    // Print visualization
    println!("\n{}", "=".repeat(80));
    println!("MODEL ARCHITECTURE VISUALIZATION");
    println!("{}", "=".repeat(80));
    println!();
    
    // Configuration
    println!("📊 MODEL CONFIGURATION");
    println!("{}", "-".repeat(40));
    println!("  Layers:           {}", architecture.config.n_layers);
    println!("  Attention heads:  {}", architecture.config.n_heads);
    println!("  KV heads:         {}", architecture.config.n_kv_heads);
    println!("  Hidden size:      {}", architecture.config.hidden_size);
    println!("  Vocab size:       {}", architecture.config.vocab_size);
    println!("  Context length:   {}", architecture.config.context_length);
    println!("  Intermediate:     {}", architecture.config.intermediate_size);
    println!("  Architecture:     {}", architecture.config.architecture);
    println!();
    
    // Embedding
    if let Some(ref emb) = architecture.embedding {
        println!("🔤 EMBEDDING LAYER");
        println!("{}", "-".repeat(40));
        println!("  Tensor: {}", emb.name);
        println!("  Shape:  {:?}", emb.dimensions);
        println!("  Type:   {:?}", emb.tensor_type);
        println!();
    }
    
    // Layers
    println!("🔗 TRANSFORMER LAYERS");
    println!("{}", "-".repeat(40));
    
    for (i, layer) in architecture.layers.iter().enumerate().take(3) {
        println!("\n  Layer {}", i);
        println!("  {}", "─".repeat(36));
        println!("  Input:  [batch, seq_len, {}]", architecture.config.hidden_size);
        println!();
        
        if layer.has_attention() {
            println!("  ┌─ Attention Block");
            
            if layer.has_component("attn_norm") {
                println!("  │  ├─ RMS Norm");
            }
            
            println!("  │  ├─ Multi-Head Attention ({} heads)", architecture.config.n_heads);
            
            if layer.has_component("q_proj") {
                let comp = &layer.components["q_proj"];
                println!("  │  │  ├─ Query:  {:?}", comp.dimensions);
            }
            if layer.has_component("k_proj") {
                let comp = &layer.components["k_proj"];
                println!("  │  │  ├─ Key:    {:?}", comp.dimensions);
            }
            if layer.has_component("v_proj") {
                let comp = &layer.components["v_proj"];
                println!("  │  │  ├─ Value:  {:?}", comp.dimensions);
            }
            
            println!("  │  │  └─ Attention: softmax(Q @ K^T) @ V");
            
            if layer.has_component("o_proj") {
                let comp = &layer.components["o_proj"];
                println!("  │  └─ Output: {:?}", comp.dimensions);
            }
            
            println!("  │  └─ Residual: input + attention_output");
            println!();
        }
        
        if layer.has_mlp() {
            println!("  └─ MLP Block");
            
            if layer.has_component("ffn_norm") {
                println!("     ├─ RMS Norm");
            }
            
            println!("     ├─ SwiGLU Activation");
            
            if layer.has_component("gate_proj") {
                let comp = &layer.components["gate_proj"];
                println!("     │  ├─ Gate:  {:?}", comp.dimensions);
            }
            if layer.has_component("up_proj") {
                let comp = &layer.components["up_proj"];
                println!("     │  ├─ Up:    {:?}", comp.dimensions);
            }
            if layer.has_component("down_proj") {
                let comp = &layer.components["down_proj"];
                println!("     │  └─ Down:  {:?}", comp.dimensions);
            }
            
            println!("     └─ Residual: residual1 + mlp_output");
        }
        
        println!();
        println!("  Output: [batch, seq_len, {}]", architecture.config.hidden_size);
    }
    
    if architecture.layers.len() > 3 {
        println!("\n  ... ({} more layers)", architecture.layers.len() - 3);
    }
    
    // Output
    if let Some(ref out) = architecture.output {
        println!("\n📤 OUTPUT LAYER");
        println!("{}", "-".repeat(40));
        println!("  Tensor: {}", out.name);
        println!("  Shape:  {:?}", out.dimensions);
        println!("  Type:   {:?}", out.tensor_type);
        println!();
    }
    
    // Statistics
    println!("📈 MODEL STATISTICS");
    println!("{}", "-".repeat(40));
    
    let total_tensors = reader.tensor_infos().len();
    let total_params: u64 = reader.tensor_infos()
        .iter()
        .map(|info| info.n_elements() as u64)
        .sum();
    
    println!("  Total tensors:    {}", total_tensors);
    println!("  Total parameters: ~{:.2}B", total_params as f64 / 1e9);
    println!();
    
    // Data flow
    println!("🌊 DATA FLOW");
    println!("{}", "-".repeat(40));
    println!();
    println!("  Text Input");
    println!("      ↓");
    println!("  Tokenization");
    println!("      ↓");
    println!("  Embedding: [{}, {}]", 
             architecture.config.vocab_size,
             architecture.config.hidden_size);
    println!("      ↓");
    
    for i in 0..std::cmp::min(3, architecture.layers.len()) {
        println!("  ┌─────────────────┐");
        println!("  │   Layer {:2}      │", i);
        println!("  │   - Attention   │");
        println!("  │   - MLP         │");
        println!("  └─────────────────┘");
        println!("      ↓");
    }
    
    if architecture.layers.len() > 3 {
        println!("      ... ({} more layers)", architecture.layers.len() - 3);
        println!("      ↓");
    }
    
    println!("  LM Head: [{}, {}]",
             architecture.config.hidden_size,
             architecture.config.vocab_size);
    println!("      ↓");
    println!("  Logits → argmax/sampling");
    println!("      ↓");
    println!("  Text Output");
    println!();
    
    Ok(())
}
