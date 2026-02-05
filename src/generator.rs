/// Lean 4 code generator with COMPLETE progress tracking
/// REMPLACER ENTIÈREMENT votre src/generator.rs par ce fichier

use crate::types::*;
use crate::reader::GGUFReader;
use std::fs::{self, File};
use std::io::{self, Write};
use std::path::{Path, PathBuf};
use std::time::Instant;

pub type Result<T> = std::result::Result<T, Box<dyn std::error::Error>>;

pub struct LeanGenerator<'a> {
    reader: &'a mut GGUFReader,
    architecture: &'a ModelArchitecture,
    output_dir: PathBuf,
    chunk_size: usize,
}

impl<'a> LeanGenerator<'a> {
    pub fn new(
        reader: &'a mut GGUFReader,
        architecture: &'a ModelArchitecture,
        output_dir: impl AsRef<Path>,
    ) -> Self {
        Self {
            reader,
            architecture,
            output_dir: output_dir.as_ref().to_path_buf(),
            chunk_size: 1000,
        }
    }
    
    /// Generate complete Lean project
    pub fn generate(&mut self) -> Result<()> {
        println!("🔨 Generating Lean project structure...");
        
        // Create directory structure
        self.create_directories()?;
        
        println!("📝 Generating base types...");
        self.generate_base_types()?;
        
        println!("⚙️  Generating operations...");
        self.generate_operations()?;
        
        println!("📦 Generating tensors...");
        self.generate_tensors()?;
        
        println!("🔗 Generating layers...");
        self.generate_layers()?;
        
        println!("🏗️  Generating model structure...");
        self.generate_model()?;
        
        println!("🎯 Generating inverse problem framework...");
        self.generate_inverse_problem()?;
        
        println!("✅ Project generated successfully!");
        Ok(())
    }
    
    fn create_directories(&self) -> Result<()> {
        fs::create_dir_all(&self.output_dir)?;
        fs::create_dir_all(self.output_dir.join("Model/Base"))?;
        fs::create_dir_all(self.output_dir.join("Model/Tensors"))?;
        fs::create_dir_all(self.output_dir.join("Model/Layers"))?;
        fs::create_dir_all(self.output_dir.join("Model/Operations"))?;
        Ok(())
    }
    
    fn generate_base_types(&self) -> Result<()> {
        let path = self.output_dir.join("Model/Base/Types.lean");
        let mut file = File::create(path)?;
        
        writeln!(file, "-- Base types for neural network representation")?;
        writeln!(file)?;
        writeln!(file, "structure Tensor where")?;
        writeln!(file, "  name : String")?;
        writeln!(file, "  dims : List Nat")?;
        writeln!(file, "  data : Array Float")?;
        writeln!(file, "  deriving Repr")?;
        writeln!(file)?;
        writeln!(file, "def shapesCompatible (dims1 dims2 : List Nat) : Bool :=")?;
        writeln!(file, "  match dims1, dims2 with")?;
        writeln!(file, "  | [m, n], [n', p] => n == n'")?;
        writeln!(file, "  | _, _ => false")?;
        writeln!(file)?;
        writeln!(file, "def isSane (t : Tensor) : Bool :=")?;
        writeln!(file, "  t.data.all (λ x => !x.isNaN && !x.isInf)")?;
        writeln!(file)?;
        writeln!(file, "opaque relu (x : Float) : Float")?;
        writeln!(file, "opaque silu (x : Float) : Float")?;
        writeln!(file, "opaque gelu (x : Float) : Float")?;
        writeln!(file, "opaque softmax (v : Array Float) : Array Float")?;
        
        Ok(())
    }
    
    fn generate_operations(&self) -> Result<()> {
        self.generate_matrix_ops()?;
        self.generate_attention_ops()?;
        self.generate_mlp_ops()?;
        Ok(())
    }
    
    fn generate_matrix_ops(&self) -> Result<()> {
        let path = self.output_dir.join("Model/Operations/Matrix.lean");
        let mut file = File::create(path)?;
        
        writeln!(file, "import Model.Base.Types")?;
        writeln!(file)?;
        writeln!(file, "opaque matmul (a b : Tensor) : Tensor")?;
        writeln!(file, "opaque elementwiseMul (a b : Tensor) : Tensor")?;
        writeln!(file, "opaque elementwiseAdd (a b : Tensor) : Tensor")?;
        
        Ok(())
    }
    
    fn generate_attention_ops(&self) -> Result<()> {
        let path = self.output_dir.join("Model/Operations/Attention.lean");
        let mut file = File::create(path)?;
        
        writeln!(file, "import Model.Base.Types")?;
        writeln!(file, "import Model.Operations.Matrix")?;
        writeln!(file)?;
        writeln!(file, "structure AttentionConfig where")?;
        writeln!(file, "  n_heads : Nat")?;
        writeln!(file, "  head_dim : Nat")?;
        writeln!(file, "  hidden_size : Nat")?;
        writeln!(file)?;
        writeln!(file, "opaque multiHeadAttention")?;
        writeln!(file, "  (config : AttentionConfig)")?;
        writeln!(file, "  (q k v : Tensor)")?;
        writeln!(file, "  (input : Tensor)")?;
        writeln!(file, "  : Tensor")?;
        
        Ok(())
    }
    
    fn generate_mlp_ops(&self) -> Result<()> {
        let path = self.output_dir.join("Model/Operations/MLP.lean");
        let mut file = File::create(path)?;
        
        writeln!(file, "import Model.Base.Types")?;
        writeln!(file, "import Model.Operations.Matrix")?;
        writeln!(file)?;
        writeln!(file, "def mlpGated (gate up down : Tensor) (input : Tensor) : Tensor :=")?;
        writeln!(file, "  let gate_out := matmul input gate")?;
        writeln!(file, "  let up_out := matmul input up")?;
        writeln!(file, "  let activated := elementwiseMul")?;
        writeln!(file, "    {{ gate_out with data := gate_out.data.map silu }}")?;
        writeln!(file, "    up_out")?;
        writeln!(file, "  matmul activated down")?;
        
        Ok(())
    }
    
    // =========================================================================
    // FONCTION PRINCIPALE AVEC PROGRESSION
    // =========================================================================
    fn generate_tensors(&mut self) -> Result<()> {
        let tensor_infos = self.reader.tensor_infos().to_vec();
        let total_tensors = tensor_infos.len();
        
        println!("   Total tensors: {}", total_tensors);
        println!();
        
        let mut generated_tensors = Vec::new();
        let start_time = Instant::now();
        
        // GÉNÈRE TOUS LES TENSORS (pas de .take())
        for (i, info) in tensor_infos.iter().enumerate() {
            // Afficher progression
            self.print_progress(i, total_tensors, info, start_time);
            
            // Générer le tensor
            let safe_name = Self::sanitize_name(&info.name);
            
            // Charger et déquantifier
            let tensor = match self.reader.load_tensor(info) {
                Ok(t) => t,
                Err(e) => {
                    println!("   ⚠️  Skipping {}: {}", info.name, e);
                    continue;
                }
            };
            
            // Vérifier sanité
            if !tensor.is_sane() {
                println!("   ⚠️  Warning: {} contains NaN/Inf values", info.name);
            }
            
            // Générer fichier Lean
            self.generate_single_tensor(&safe_name, &tensor, info)?;
            generated_tensors.push(safe_name);
        }
        
        let total_time = start_time.elapsed();
        
        // Résumé final
        println!();
        println!("✅ Tensor generation complete!");
        println!("   Generated: {} tensors", generated_tensors.len());
        println!("   Total time: {:.1}s ({:.1} min)", 
                 total_time.as_secs_f32(), 
                 total_time.as_secs_f32() / 60.0);
        println!("   Average: {:.1}s per tensor", 
                 total_time.as_secs_f32() / generated_tensors.len() as f32);
        
        // Générer le fichier d'index
        self.generate_tensor_index(&generated_tensors)?;
        
        Ok(())
    }
    
    // NOUVELLE FONCTION : Affichage de progression
    fn print_progress(&self, current: usize, total: usize, info: &TensorInfo, start_time: Instant) {
        let percentage = ((current + 1) as f32 / total as f32 * 100.0) as u32;
        let elapsed = start_time.elapsed().as_secs_f32();
        
        // Calculer temps restant estimé
        let avg_time_per_tensor = elapsed / (current + 1) as f32;
        let remaining_tensors = total - (current + 1);
        let eta_seconds = avg_time_per_tensor * remaining_tensors as f32;
        
        // Formater ETA
        let eta_str = Self::format_eta(eta_seconds);
        
        // Barre de progression
        let bar_width = 40;
        let filled = (bar_width as f32 * (current + 1) as f32 / total as f32) as usize;
        let bar: String = (0..bar_width)
            .map(|i| if i < filled { '█' } else { '░' })
            .collect();
        
        // Taille du tensor
        let size_mb = (info.n_elements() * 4) as f32 / 1024.0 / 1024.0;
        
        // Afficher avec \r pour réécrire sur la même ligne
        print!("\r   [{:4}/{:4}] [{}] {:3}% | {:35} | {:8.1} MB | ETA: {:>8}",
            current + 1,
            total,
            bar,
            percentage,
            Self::truncate_name(&info.name, 35),
            size_mb,
            eta_str
        );
        
        // Forcer l'affichage immédiat
        io::stdout().flush().unwrap();
        
        // Nouvelle ligne tous les 10 tensors
        if (current + 1) % 10 == 0 {
            println!();
        }
    }
    
    // NOUVELLE FONCTION : Formater ETA
    fn format_eta(seconds: f32) -> String {
        if seconds < 60.0 {
            format!("{:.0}s", seconds)
        } else if seconds < 3600.0 {
            let mins = (seconds / 60.0) as u32;
            let secs = (seconds % 60.0) as u32;
            format!("{}m {:02}s", mins, secs)
        } else {
            let hours = (seconds / 3600.0) as u32;
            let mins = ((seconds % 3600.0) / 60.0) as u32;
            format!("{}h {:02}m", hours, mins)
        }
    }
    
    // NOUVELLE FONCTION : Tronquer noms longs
    fn truncate_name(name: &str, max_len: usize) -> String {
        if name.len() <= max_len {
            format!("{:<width$}", name, width = max_len)
        } else {
            let start_len = max_len - 5;
            format!("{}..{}", &name[..start_len], &name[name.len()-2..])
        }
    }
    
    fn generate_single_tensor(&self, safe_name: &str, tensor: &Tensor, info: &TensorInfo) -> Result<()> {
        let path = self.output_dir.join(format!("Model/Tensors/{}.lean", safe_name));
        let mut file = File::create(path)?;
        
        writeln!(file, "import Model.Base.Types")?;
        writeln!(file)?;
        writeln!(file, "-- Tensor: {}", info.name)?;
        writeln!(file, "-- Dimensions: {:?}", info.dimensions)?;
        writeln!(file, "-- Type: {:?}", info.tensor_type)?;
        writeln!(file)?;
        
        // Generate data in chunks
        writeln!(file, "def {}_data : Array Float := ", safe_name)?;
        
        let data = &tensor.data;
        for (chunk_idx, chunk) in data.chunks(self.chunk_size).enumerate() {
            write!(file, "  ")?;
            if chunk_idx == 0 {
                write!(file, "#[")?;
            }
            
            for (i, val) in chunk.iter().enumerate() {
                if i > 0 {
                    write!(file, ", ")?;
                }
                write!(file, "{}", val)?;
            }
            
            if chunk_idx == data.chunks(self.chunk_size).count() - 1 {
                writeln!(file, "]")?;
            } else {
                writeln!(file, ",")?;
            }
        }
        
        writeln!(file)?;
        writeln!(file, "def {} : Tensor := {{", safe_name)?;
        writeln!(file, "  name := \"{}\"", info.name)?;
        writeln!(file, "  dims := {:?}", info.dimensions)?;
        writeln!(file, "  data := {}_data", safe_name)?;
        writeln!(file, "}}")?;
        
        Ok(())
    }
    
    fn generate_tensor_index(&self, tensor_names: &[String]) -> Result<()> {
        let path = self.output_dir.join("Model/Tensors.lean");
        let mut file = File::create(path)?;
        
        writeln!(file, "-- Auto-generated tensor index")?;
        writeln!(file)?;
        for name in tensor_names {
            writeln!(file, "import Model.Tensors.{}", name)?;
        }
        
        Ok(())
    }
    
    fn generate_layers(&self) -> Result<()> {
        // Simplified layer generation
        let path = self.output_dir.join("Model/Layers.lean");
        let mut file = File::create(path)?;
        
        writeln!(file, "import Model.Base.Types")?;
        writeln!(file, "import Model.Operations.Attention")?;
        writeln!(file, "import Model.Operations.MLP")?;
        writeln!(file)?;
        writeln!(file, "-- Layer definitions go here")?;
        
        Ok(())
    }
    
    fn generate_model(&self) -> Result<()> {
        let path = self.output_dir.join("Model.lean");
        let mut file = File::create(path)?;
        
        writeln!(file, "import Model.Base.Types")?;
        writeln!(file, "import Model.Tensors")?;
        writeln!(file, "import Model.Layers")?;
        writeln!(file)?;
        writeln!(file, "-- Model configuration")?;
        writeln!(file, "def n_layers : Nat := {}", self.architecture.config.n_layers)?;
        writeln!(file, "def hidden_size : Nat := {}", self.architecture.config.hidden_size)?;
        writeln!(file, "def vocab_size : Nat := {}", self.architecture.config.vocab_size)?;
        
        Ok(())
    }
    
    fn generate_inverse_problem(&self) -> Result<()> {
        let path = self.output_dir.join("InverseProblem.lean");
        let mut file = File::create(path)?;
        
        writeln!(file, "import Model")?;
        writeln!(file)?;
        writeln!(file, "-- Inverse problem: find input that produces target output")?;
        writeln!(file, "axiom ProducesOutput : String → String → Prop")?;
        writeln!(file)?;
        writeln!(file, "theorem inverse_problem_exists (target : String) :")?;
        writeln!(file, "  ∃ input, ProducesOutput input target := by")?;
        writeln!(file, "  sorry")?;
        
        Ok(())
    }
    
    fn sanitize_name(name: &str) -> String {
        name.replace(".", "_")
            .replace("-", "_")
            .replace("/", "_")
    }
}
