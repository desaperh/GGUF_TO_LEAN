/**
Copyright (C) 2021 - Hugo DE SA PEREIRA PINTO ( hugo.de.sa.pereira@gmail.com)

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

/// Command-line interface for GGUF to Lean converter
use std::env;
use std::process;

// Import des modules locaux directement
mod types;
mod reader;
mod analyzer;
mod generator;

use reader::GGUFReader;
use analyzer::ArchitectureAnalyzer;
use generator::LeanGenerator;

fn main() {
    let args: Vec<String> = env::args().collect();
    
    if args.len() < 3 {
        eprintln!("Usage: {} <input.gguf> <output_dir>", args[0]);
        eprintln!("\nThis will generate a complete Lean 4 project with:");
        eprintln!("  - Full model architecture (layers, attention, MLP)");
        eprintln!("  - Tensor definitions with sanity checks");
        eprintln!("  - Operations (matmul, attention, activations)");
        eprintln!("  - Inverse problem framework for finding inputs");
        process::exit(1);
    }
    
    let gguf_path = &args[1];
    let output_dir = &args[2];
    
    if let Err(e) = convert_gguf_to_lean(gguf_path, output_dir) {
        eprintln!("Error: {}", e);
        process::exit(1);
    }
}

/// Main conversion pipeline
fn convert_gguf_to_lean(
    gguf_path: &str,
    output_dir: &str,
) -> Result<(), Box<dyn std::error::Error>> {
    println!("📂 Reading GGUF file: {}", gguf_path);
    
    // Parse GGUF file
    let mut reader = GGUFReader::open(gguf_path)?;
    reader.parse()?;
    
    println!("🔍 Analyzing architecture...");
    
    // Analyze architecture
    let analyzer = ArchitectureAnalyzer::new(
        reader.tensor_infos(),
        reader.metadata(),
    );
    let architecture = analyzer.analyze();
    
    // Print architecture info
    println!("📊 Model info:");
    println!("  - Layers: {}", architecture.config.n_layers);
    println!("  - Hidden size: {}", architecture.config.hidden_size);
    println!("  - Vocab size: {}", architecture.config.vocab_size);
    println!("  - Total tensors: {}", reader.tensor_infos().len());
    println!("  - Architecture: {}", architecture.config.architecture);
    
    // Generate Lean code
    println!("\n🔨 Generating Lean project...");
    let mut generator = LeanGenerator::new(&mut reader, &architecture, output_dir);
    generator.generate()?;
    
    println!("\n✅ Conversion complete!");
    println!("   Output directory: {}", output_dir);
    println!("\nNext steps:");
    println!("  1. cd {}", output_dir);
    println!("  2. Edit Search.lean to define your target output");
    println!("  3. Use Lean tactics to prove the existence theorem");
    
    Ok(())
}
