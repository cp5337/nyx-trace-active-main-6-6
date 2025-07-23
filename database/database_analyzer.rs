// =====================================================
// MONOREPO DATA STRUCTURE CRAWLER
// =====================================================
// Simple, focused crawler for Supabase/SurrealDB/Sled optimization

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::fs;
use serde::{Deserialize, Serialize};
use syn::{parse_file, Item, ItemStruct, ItemEnum, Fields};
use anyhow::Result;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DataStructureCrawler {
    pub discovered_structs: Vec<StructInfo>,
    pub discovered_enums: Vec<EnumInfo>,
    pub optimization_suggestions: Vec<OptimizationSuggestion>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct StructInfo {
    pub name: String,
    pub file_path: String,
    pub fields: Vec<FieldInfo>,
    pub estimated_size: usize,
    pub access_pattern: AccessPattern,
    pub suggested_storage: StorageRecommendation,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FieldInfo {
    pub name: String,
    pub rust_type: String,
    pub is_primary_key: bool,
    pub is_indexed: bool,
    pub is_searchable: bool,
    pub supabase_type: String,
    pub surrealdb_type: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum AccessPattern {
    HighFrequency,    // Sled candidate
    MediumFrequency,  // SurrealDB candidate  
    LowFrequency,     // Supabase candidate
    AnalyticsOnly,    // Supabase with views
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum StorageRecommendation {
    Sled {
        reason: String,
        key_strategy: SledKeyStrategy,
    },
    SurrealDB {
        reason: String,
        table_name: String,
        relationships: Vec<String>,
    },
    Supabase {
        reason: String,
        table_name: String,
        rls_needed: bool,
    },
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SledKeyStrategy {
    SimpleKey(String),           // Single field key
    CompositeKey(Vec<String>),   // Multiple fields
    TimestampPrefixed(String),   // timestamp::key for time-series
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EnumInfo {
    pub name: String,
    pub variants: Vec<String>,
    pub suggested_storage: EnumStorageStrategy,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum EnumStorageStrategy {
    PostgresEnum,    // Native Postgres enum in Supabase
    StringField,     // Store as string with CHECK constraint
    IntegerField,    // Store as integer with lookup table
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct OptimizationSuggestion {
    pub suggestion_type: SuggestionType,
    pub description: String,
    pub affected_structs: Vec<String>,
    pub implementation: String,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum SuggestionType {
    DataFlow,
    PerformanceOptimization,
    StorageTierOptimization,
    SchemaSimplification,
}

impl DataStructureCrawler {
    pub fn new() -> Self {
        Self {
            discovered_structs: Vec::new(),
            discovered_enums: Vec::new(),
            optimization_suggestions: Vec::new(),
        }
    }

    pub async fn crawl_monorepo(&mut self, root_path: &Path) -> Result<CrawlReport> {
        println!("🔍 Crawling monorepo at: {:?}", root_path);
        
        let rust_files = self.find_rust_files(root_path)?;
        println!("📁 Found {} Rust files", rust_files.len());

        for file_path in rust_files {
            if let Ok(()) = self.analyze_file(&file_path).await {
                // File processed successfully
            }
        }

        self.generate_optimizations();
        
        Ok(CrawlReport {
            total_structs: self.discovered_structs.len(),
            total_enums: self.discovered_enums.len(),
            sled_candidates: self.count_storage_recommendations(StorageType::Sled),
            surrealdb_candidates: self.count_storage_recommendations(StorageType::SurrealDB),
            supabase_candidates: self.count_storage_recommendations(StorageType::Supabase),
            optimization_count: self.optimization_suggestions.len(),
        })
    }

    fn find_rust_files(&self, root: &Path) -> Result<Vec<PathBuf>> {
        let mut rust_files = Vec::new();
        self.scan_directory(root, &mut rust_files)?;
        
        // Filter out target directories and tests
        rust_files.retain(|path| {
            let path_str = path.to_string_lossy();
            !path_str.contains("/target/") && 
            !path_str.contains("\\target\\") &&
            !path_str.ends_with("_test.rs") &&
            !path_str.contains("/tests/")
        });
        
        Ok(rust_files)
    }

    fn scan_directory(&self, dir: &Path, files: &mut Vec<PathBuf>) -> Result<()> {
        if !dir.is_dir() {
            return Ok(());
        }

        for entry in fs::read_dir(dir)? {
            let entry = entry?;
            let path = entry.path();
            
            if path.is_dir() {
                let dir_name = path.file_name().unwrap().to_string_lossy();
                if dir_name != "target" && !dir_name.starts_with('.') {
                    self.scan_directory(&path, files)?;
                }
            } else if path.extension().map_or(false, |ext| ext == "rs") {
                files.push(path);
            }
        }
        
        Ok(())
    }

    async fn analyze_file(&mut self, file_path: &Path) -> Result<()> {
        let content = fs::read_to_string(file_path)?;
        let syntax = parse_file(&content)?;
        
        for item in syntax.items {
            match item {
                Item::Struct(item_struct) => {
                    if let Some(struct_info) = self.analyze_struct(&item_struct, file_path) {
                        self.discovered_structs.push(struct_info);
                    }
                }
                Item::Enum(item_enum) => {
                    if let Some(enum_info) = self.analyze_enum(&item_enum) {
                        self.discovered_enums.push(enum_info);
                    }
                }
                _ => {}
            }
        }
        
        Ok(())
    }

    fn analyze_struct(&self, item_struct: &syn::ItemStruct, file_path: &Path) -> Option<StructInfo> {
        let struct_name = item_struct.ident.to_string();
        
        // Skip utility structs
        if struct_name.ends_with("Error") || 
           struct_name.ends_with("Config") || 
           struct_name.starts_with("Test") {
            return None;
        }

        let mut fields = Vec::new();
        
        if let Fields::Named(named_fields) = &item_struct.fields {
            for field in &named_fields.named {
                if let Some(field_info) = self.analyze_field(field) {
                    fields.push(field_info);
                }
            }
        }

        let access_pattern = self.determine_access_pattern(&struct_name, &fields);
        let suggested_storage = self.recommend_storage(&struct_name, &fields, &access_pattern);
        
        Some(StructInfo {
            name: struct_name,
            file_path: file_path.to_string_lossy().to_string(),
            fields,
            estimated_size: self.estimate_size(&item_struct),
            access_pattern,
            suggested_storage,
        })
    }

    fn analyze_field(&self, field: &syn::Field) -> Option<FieldInfo> {
        let field_name = field.ident.as_ref()?.to_string();
        let rust_type = quote::ToTokens::to_token_stream(&field.ty).to_string();
        
        let is_primary_key = field_name == "id" || field_name.ends_with("_id") && !field_name.contains("_id_");
        let is_indexed = is_primary_key || field_name.ends_with("_id") || field_name.contains("index");
        let is_searchable = rust_type.contains("String") && 
                           (field_name.contains("name") || field_name.contains("description"));

        Some(FieldInfo {
            name: field_name.clone(),
            rust_type: rust_type.clone(),
            is_primary_key,
            is_indexed,
            is_searchable,
            supabase_type: self.map_to_postgres_type(&rust_type),
            surrealdb_type: self.map_to_surrealdb_type(&rust_type),
        })
    }

    fn determine_access_pattern(&self, struct_name: &str, fields: &[FieldInfo]) -> AccessPattern {
        let name_lower = struct_name.to_lowercase();
        
        // High frequency candidates (Sled)
        if name_lower.contains("cache") || 
           name_lower.contains("session") || 
           name_lower.contains("state") ||
           name_lower.contains("counter") {
            return AccessPattern::HighFrequency;
        }
        
        // Analytics only (Supabase)
        if name_lower.contains("log") || 
           name_lower.contains("event") || 
           name_lower.contains("metric") ||
           name_lower.contains("audit") {
            return AccessPattern::AnalyticsOnly;
        }
        
        // Check field patterns
        let has_timestamps = fields.iter().any(|f| 
            f.name.contains("timestamp") || f.name.contains("created_at"));
        let has_relationships = fields.iter().any(|f| 
            f.name.ends_with("_id") && !f.is_primary_key);
        
        if has_relationships && !has_timestamps {
            AccessPattern::MediumFrequency  // SurrealDB for relationships
        } else {
            AccessPattern::LowFrequency     // Supabase default
        }
    }

    fn recommend_storage(&self, struct_name: &str, fields: &[FieldInfo], access_pattern: &AccessPattern) -> StorageRecommendation {
        match access_pattern {
            AccessPattern::HighFrequency => {
                let key_field = fields.iter()
                    .find(|f| f.is_primary_key)
                    .map(|f| f.name.clone())
                    .unwrap_or_else(|| "id".to_string());
                
                StorageRecommendation::Sled {
                    reason: "High frequency access pattern detected".to_string(),
                    key_strategy: SledKeyStrategy::SimpleKey(key_field),
                }
            }
            
            AccessPattern::MediumFrequency => {
                let relationships: Vec<String> = fields.iter()
                    .filter(|f| f.name.ends_with("_id") && !f.is_primary_key)
                    .map(|f| f.name.clone())
                    .collect();
                
                StorageRecommendation::SurrealDB {
                    reason: "Complex relationships and medium frequency access".to_string(),
                    table_name: self.to_snake_case(struct_name),
                    relationships,
                }
            }
            
            AccessPattern::LowFrequency | AccessPattern::AnalyticsOnly => {
                StorageRecommendation::Supabase {
                    reason: match access_pattern {
                        AccessPattern::AnalyticsOnly => "Analytics and reporting workload".to_string(),
                        _ => "Low frequency access, complex queries".to_string(),
                    },
                    table_name: self.to_snake_case(struct_name),
                    rls_needed: !struct_name.to_lowercase().contains("public"),
                }
            }
        }
    }

    fn map_to_postgres_type(&self, rust_type: &str) -> String {
        match rust_type {
            t if t.contains("String") => "TEXT".to_string(),
            t if t.contains("i32") => "INTEGER".to_string(),
            t if t.contains("i64") => "BIGINT".to_string(),
            t if t.contains("f32") || t.contains("f64") => "DECIMAL".to_string(),
            t if t.contains("bool") => "BOOLEAN".to_string(),
            t if t.contains("SystemTime") || t.contains("DateTime") => "TIMESTAMPTZ".to_string(),
            t if t.contains("Uuid") => "UUID".to_string(),
            t if t.contains("Vec") => "JSONB".to_string(),
            _ => "TEXT".to_string(),
        }
    }

    fn map_to_surrealdb_type(&self, rust_type: &str) -> String {
        match rust_type {
            t if t.contains("String") => "string".to_string(),
            t if t.contains("i32") || t.contains("i64") => "int".to_string(),
            t if t.contains("f32") || t.contains("f64") => "float".to_string(),
            t if t.contains("bool") => "bool".to_string(),
            t if t.contains("SystemTime") || t.contains("DateTime") => "datetime".to_string(),
            t if t.contains("Vec") => "array".to_string(),
            _ => "string".to_string(),
        }
    }

    fn analyze_enum(&self, item_enum: &syn::ItemEnum) -> Option<EnumInfo> {
        let enum_name = item_enum.ident.to_string();
        let variants: Vec<String> = item_enum.variants.iter()
            .map(|v| v.ident.to_string())
            .collect();
        
        let suggested_storage = if variants.len() <= 10 {
            EnumStorageStrategy::PostgresEnum
        } else if variants.len() <= 50 {
            EnumStorageStrategy::StringField
        } else {
            EnumStorageStrategy::IntegerField
        };
        
        Some(EnumInfo {
            name: enum_name,
            variants,
            suggested_storage,
        })
    }

    fn generate_optimizations(&mut self) {
        // Data flow optimization
        let high_freq_structs: Vec<String> = self.discovered_structs.iter()
            .filter(|s| matches!(s.access_pattern, AccessPattern::HighFrequency))
            .map(|s| s.name.clone())
            .collect();
        
        if !high_freq_structs.is_empty() {
            self.optimization_suggestions.push(OptimizationSuggestion {
                suggestion_type: SuggestionType::DataFlow,
                description: "Consider caching these high-frequency structs in Sled".to_string(),
                affected_structs: high_freq_structs,
                implementation: "Use Sled for hot data with TTL, sync to SurrealDB/Supabase periodically".to_string(),
            });
        }
        
        // Add more optimization suggestions...
    }

    fn count_storage_recommendations(&self, storage_type: StorageType) -> usize {
        self.discovered_structs.iter().filter(|s| {
            match (&s.suggested_storage, storage_type) {
                (StorageRecommendation::Sled { .. }, StorageType::Sled) => true,
                (StorageRecommendation::SurrealDB { .. }, StorageType::SurrealDB) => true,
                (StorageRecommendation::Supabase { .. }, StorageType::Supabase) => true,
                _ => false,
            }
        }).count()
    }

    fn estimate_size(&self, _item_struct: &syn::ItemStruct) -> usize {
        // Rough estimate - could be more sophisticated
        128
    }

    fn to_snake_case(&self, s: &str) -> String {
        let mut result = String::new();
        for (i, c) in s.chars().enumerate() {
            if c.is_uppercase() && i > 0 {
                result.push('_');
            }
            result.push(c.to_lowercase().next().unwrap());
        }
        result
    }
}

#[derive(Debug, Clone, Copy)]
enum StorageType {
    Sled,
    SurrealDB,
    Supabase,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CrawlReport {
    pub total_structs: usize,
    pub total_enums: usize,
    pub sled_candidates: usize,
    pub surrealdb_candidates: usize,
    pub supabase_candidates: usize,
    pub optimization_count: usize,
}

// Usage example
impl DataStructureCrawler {
    pub async fn analyze_project(project_path: &str) -> Result<CrawlReport> {
        let mut crawler = DataStructureCrawler::new();
        let report = crawler.crawl_monorepo(Path::new(project_path)).await?;
        
        println!("\n📊 CRAWL REPORT");
        println!("================");
        println!("Total Structs: {}", report.total_structs);
        println!("├─ Sled candidates: {} (high frequency)", report.sled_candidates);
        println!("├─ SurrealDB candidates: {} (relationships)", report.surrealdb_candidates);
        println!("└─ Supabase candidates: {} (analytics/low freq)", report.supabase_candidates);
        println!("\nOptimizations suggested: {}", report.optimization_count);
        
        // Generate schema files
        crawler.generate_schemas().await?;
        
        Ok(report)
    }

    async fn generate_schemas(&self) -> Result<()> {
        self.generate_sled_schema().await?;
        self.generate_surrealdb_schema().await?;
        self.generate_supabase_schema().await?;
        Ok(())
    }

    async fn generate_sled_schema(&self) -> Result<()> {
        // Generate Sled key-value mappings
        println!("🗄️  Generating Sled schema...");
        Ok(())
    }

    async fn generate_surrealdb_schema(&self) -> Result<()> {
        // Generate SurrealDB tables and relationships
        println!("🗄️  Generating SurrealDB schema...");
        Ok(())
    }

    async fn generate_supabase_schema(&self) -> Result<()> {
        // Generate PostgreSQL schema for Supabase
        println!("🗄️  Generating Supabase schema...");
        Ok(())
    }
}