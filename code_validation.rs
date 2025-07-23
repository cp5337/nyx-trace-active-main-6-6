// =====================================================
// CTAS ENHANCED MONOREPO SCANNER v6.6
// =====================================================
// Dr. Adelaide V. Langford Integration - "QuantaKnife" Standards
// DISA-compliant validation with persona attribution and USIM compliance
// hashid: ctas_enhanced_scanner_001a2b3c4d5e6f7g8h9i0j1k
// cuid: c1m6k9p4q7r2s5t8u9v0w1x2y3z4a5b6c7d8e9f0g1h2i3j4k5l6
// ttl: 86400

use std::fs::{self, DirEntry};
use std::path::{Path, PathBuf};
use std::io::Write;
use std::collections::HashMap;
use serde::{Deserialize, Serialize};
use chrono::{DateTime, Utc};
use sha2::{Sha256, Digest};
use regex::Regex;

// RDF: CTASEnhancedScanner rdf:type ctas:ValidationSystem
// RDF: LangfordIntegration ctas:validates ctas:MonorepoStructure
// RDF: PersonaAttribution ctas:enforces ctas:EliteTeamStandards
// RDF: USIMCompliance ctas:validates ctas:HashCUIDTTL

/// Enhanced file status with Dr. Langford's validation criteria
#[derive(Debug, Serialize, Deserialize)]
struct EnhancedFileStatus {
    path: String,
    file_type: FileType,
    is_complete: bool,
    langford_validation: LangfordValidation,
    persona_attribution: PersonaAttribution,
    usim_compliance: USIMCompliance,
    rdf_compliance: RDFCompliance,
    code_quality: CodeQuality,
    security_compliance: SecurityCompliance,
}

#[derive(Debug, Serialize, Deserialize)]
enum FileType {
    RustSource,
    PythonSource,
    TypeScriptSource,
    JavaScriptSource,
    ShellScript,
    ConfigFile,
    DocumentationFile,
    TestFile,
    Unknown,
}

#[derive(Debug, Serialize, Deserialize)]
struct LangfordValidation {
    inspector: String,              // "Dr. Adelaide V. Langford"
    validation_passed: bool,
    compliance_score: f64,
    crypto_signature: String,       // "QK-L1-{HASH}-AUTH-HEX"
    issues: Vec<ValidationIssue>,
    disa_clearance: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct PersonaAttribution {
    assigned_persona: Option<String>,
    persona_confidence: f64,
    elite_team_member: bool,
    persona_keywords: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct USIMCompliance {
    has_hashid: bool,
    has_cuid: bool,
    has_ttl: bool,
    hashid_valid: bool,
    cuid_valid: bool,
    ttl_valid: bool,
    compliance_score: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct RDFCompliance {
    has_rdf_comments: bool,
    comment_density: f64,
    triplet_format_valid: bool,
    min_density_met: bool,      // 30% requirement
}

#[derive(Debug, Serialize, Deserialize)]
struct CodeQuality {
    line_count: usize,
    function_count: usize,
    complexity_score: f64,
    has_tests: bool,
    documentation_coverage: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct SecurityCompliance {
    no_hardcoded_secrets: bool,
    no_dangerous_patterns: bool,
    crypto_usage_safe: bool,
    input_validation_present: bool,
}

#[derive(Debug, Serialize, Deserialize)]
struct ValidationIssue {
    category: String,
    severity: String,
    description: String,
    line_number: Option<u32>,
    remediation: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct CTASValidationReport {
    metadata: ReportMetadata,
    summary: ValidationSummary,
    files: Vec<EnhancedFileStatus>,
    langford_verdict: LangfordOverallVerdict,
    elite_team_status: EliteTeamStatus,
    recommendations: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct ReportMetadata {
    inspector: String,
    scan_timestamp: DateTime<Utc>,
    repo_path: String,
    total_files_scanned: usize,
    scan_duration_ms: u128,
    ctas_version: String,
}

#[derive(Debug, Serialize, Deserialize)]
struct ValidationSummary {
    total_code_files: usize,
    complete_files: usize,
    incomplete_files: usize,
    persona_assigned_files: usize,
    usim_compliant_files: usize,
    rdf_compliant_files: usize,
    overall_compliance_score: f64,
}

#[derive(Debug, Serialize, Deserialize)]
struct LangfordOverallVerdict {
    final_result: String,           // APPROVED, CONDITIONAL, REJECTED, CRITICAL_FAILURE
    overall_score: f64,
    crypto_signature: String,
    authority_statement: String,
    critical_issues: Vec<String>,
}

#[derive(Debug, Serialize, Deserialize)]
struct EliteTeamStatus {
    natasha_volkov_files: usize,    // Technical Lead & AI Specialist
    omar_al_rashid_files: usize,    // MENA Operations & Cultural Intelligence
    james_sterling_files: usize,    // Financial Intelligence & Sanctions Analysis
    emily_chen_files: usize,        // Deception Operations & Cyber Warfare
    dmitri_kozlov_files: usize,     // Advanced Persistent Threat Analysis
    aria_nakamura_files: usize,     // AI Security & Neural Network Defense
    adelaide_langford_files: usize, // QA & Security Validation Lead
    unassigned_files: usize,
}

/// Enhanced CTAS Monorepo Scanner with Dr. Langford Integration
pub struct CTASEnhancedScanner {
    ignore_patterns: Vec<String>,
    elite_personas: HashMap<String, Vec<String>>,
    validation_rules: ValidationRules,
}

#[derive(Debug)]
struct ValidationRules {
    min_rdf_density: f64,           // 30%
    required_usim_headers: Vec<String>,
    dangerous_patterns: Vec<String>,
    persona_keywords: HashMap<String, Vec<String>>,
}

impl CTASEnhancedScanner {
    pub fn new() -> Self {
        let mut elite_personas = HashMap::new();
        
        // Elite team persona keyword mapping
        elite_personas.insert("Natasha Volkov".to_string(), vec![
            "cognigraph".to_string(), "neural".to_string(), "sch_engine".to_string(),
            "ai".to_string(), "lattice".to_string(), "repo_prompt".to_string()
        ]);
        
        elite_personas.insert("Omar Al-Rashid".to_string(), vec![
            "deception".to_string(), "cultural".to_string(), "mena".to_string(),
            "osint".to_string(), "arabic".to_string(), "middle_east".to_string()
        ]);
        
        elite_personas.insert("James Sterling".to_string(), vec![
            "financial".to_string(), "crypto".to_string(), "blockchain".to_string(),
            "sanctions".to_string(), "economic".to_string(), "intelligence".to_string()
        ]);
        
        elite_personas.insert("Emily Chen".to_string(), vec![
            "deception".to_string(), "cyber".to_string(), "warfare".to_string(),
            "infrastructure".to_string(), "digital_twin".to_string(), "hospital".to_string()
        ]);
        
        elite_personas.insert("Dmitri Kozlov".to_string(), vec![
            "apt".to_string(), "threat".to_string(), "malware".to_string(),
            "russian".to_string(), "lazarus".to_string(), "analysis".to_string()
        ]);
        
        elite_personas.insert("Aria Nakamura".to_string(), vec![
            "ai_security".to_string(), "neural".to_string(), "model".to_string(),
            "transformer".to_string(), "backdoor".to_string(), "poison".to_string()
        ]);
        
        elite_personas.insert("Dr. Adelaide V. Langford".to_string(), vec![
            "validation".to_string(), "qa".to_string(), "audit".to_string(),
            "compliance".to_string(), "disa".to_string(), "security".to_string()
        ]);

        Self {
            ignore_patterns: Self::load_ignore_patterns(),
            elite_personas,
            validation_rules: ValidationRules {
                min_rdf_density: 30.0,
                required_usim_headers: vec!["hashid".to_string(), "cuid".to_string(), "ttl".to_string()],
                dangerous_patterns: vec![
                    "eval(".to_string(),
                    "innerHTML".to_string(),
                    "dangerouslySetInnerHTML".to_string(),
                    "document.write".to_string(),
                ],
                persona_keywords: elite_personas.clone(),
            },
        }
    }

    /// Load .ctas-ignore patterns for Dr. Langford's validation rules
    fn load_ignore_patterns() -> Vec<String> {
        let ignore_path = Path::new(".ctas-ignore");
        if ignore_path.exists() {
            fs::read_to_string(ignore_path)
                .unwrap_or_default()
                .lines()
                .filter(|line| !line.trim().is_empty() && !line.starts_with('#'))
                .map(|line| line.to_string())
                .collect()
        } else {
            // Default ignore patterns
            vec![
                "target/".to_string(),
                "node_modules/".to_string(),
                ".git/".to_string(),
                "*.tmp".to_string(),
                "*.log".to_string(),
            ]
        }
    }

    /// Check if file should be ignored based on .ctas-ignore patterns
    fn should_ignore(&self, path: &Path) -> bool {
        let path_str = path.to_string_lossy();
        
        for pattern in &self.ignore_patterns {
            if pattern.ends_with('/') {
                // Directory pattern
                if path_str.contains(pattern) {
                    return true;
                }
            } else if pattern.contains('*') {
                // Glob pattern (simple implementation)
                let regex_pattern = pattern.replace("*", ".*");
                if let Ok(regex) = Regex::new(&regex_pattern) {
                    if regex.is_match(&path_str) {
                        return true;
                    }
                }
            } else {
                // Exact match
                if path_str.contains(pattern) {
                    return true;
                }
            }
        }
        
        false
    }

    /// Determine file type based on extension and content
    fn determine_file_type(&self, path: &Path) -> FileType {
        if let Some(extension) = path.extension() {
            match extension.to_string_lossy().as_ref() {
                "rs" => FileType::RustSource,
                "py" => FileType::PythonSource,
                "ts" | "tsx" => FileType::TypeScriptSource,
                "js" | "jsx" => FileType::JavaScriptSource,
                "sh" | "bash" => FileType::ShellScript,
                "json" | "yaml" | "yml" | "toml" => FileType::ConfigFile,
                "md" | "txt" | "rst" => FileType::DocumentationFile,
                _ => {
                    // Check if it's a test file
                    let file_name = path.file_name().unwrap().to_string_lossy();
                    if file_name.contains("test") || file_name.contains("spec") {
                        FileType::TestFile
                    } else {
                        FileType::Unknown
                    }
                }
            }
        } else {
            FileType::Unknown
        }
    }

    /// Enhanced code completion check
    fn is_code_complete(&self, contents: &str, file_type: &FileType) -> bool {
        let lines: Vec<&str> = contents.lines().collect();
        if lines.is_empty() {
            return false;
        }

        let last_non_empty_line = lines.iter().rev().find(|l| !l.trim().is_empty());
        
        match last_non_empty_line {
            Some(line) => {
                let trimmed = line.trim();
                
                // File type specific checks
                match file_type {
                    FileType::RustSource => {
                        !trimmed.ends_with('{') && 
                        !trimmed.ends_with("fn") && 
                        !trimmed.ends_with("struct") &&
                        !trimmed.ends_with("impl") &&
                        !trimmed.ends_with("trait") &&
                        !trimmed.contains("// TODO") &&
                        !trimmed.contains("unimplemented!")
                    },
                    FileType::PythonSource => {
                        !trimmed.ends_with(':') &&
                        !trimmed.ends_with("def") &&
                        !trimmed.ends_with("class") &&
                        !trimmed.contains("# TODO") &&
                        !trimmed.contains("pass  # placeholder")
                    },
                    FileType::TypeScriptSource => {
                        !trimmed.ends_with('{') &&
                        !trimmed.ends_with("function") &&
                        !trimmed.ends_with("interface") &&
                        !trimmed.contains("// TODO")
                    },
                    _ => !trimmed.ends_with('{') && !trimmed.ends_with(':'),
                }
            },
            None => false,
        }
    }

    /// Validate USIM headers (hashid, cuid, ttl) as per Dr. Langford's standards
    fn validate_usim_compliance(&self, contents: &str) -> USIMCompliance {
        let hashid_regex = Regex::new(r"hashid:\s*([a-f0-9]{8,64})").unwrap();
        let cuid_regex = Regex::new(r"cuid:\s*([a-z0-9]{20,32})").unwrap();
        let ttl_regex = Regex::new(r"ttl:\s*(\d+)").unwrap();

        let has_hashid = contents.contains("hashid:");
        let has_cuid = contents.contains("cuid:");
        let has_ttl = contents.contains("ttl:");

        let hashid_valid = hashid_regex.is_match(contents);
        let cuid_valid = cuid_regex.is_match(contents);
        let ttl_valid = ttl_regex.is_match(contents);

        let compliance_score = [has_hashid, has_cuid, has_ttl, hashid_valid, cuid_valid, ttl_valid]
            .iter()
            .map(|&b| if b { 1.0 } else { 0.0 })
            .sum::<f64>() / 6.0 * 100.0;

        USIMCompliance {
            has_hashid,
            has_cuid,
            has_ttl,
            hashid_valid,
            cuid_valid,
            ttl_valid,
            compliance_score,
        }
    }

    /// Validate RDF comment compliance (30% density requirement)
    fn validate_rdf_compliance(&self, contents: &str) -> RDFCompliance {
        let lines: Vec<&str> = contents.lines().collect();
        let total_lines = lines.len();
        
        if total_lines == 0 {
            return RDFCompliance {
                has_rdf_comments: false,
                comment_density: 0.0,
                triplet_format_valid: false,
                min_density_met: false,
            };
        }

        let comment_lines = lines.iter()
            .filter(|line| line.trim().starts_with("//") || line.trim().starts_with("/*"))
            .count();

        let rdf_lines = lines.iter()
            .filter(|line| line.contains("// RDF:") || line.contains("/* RDF:"))
            .count();

        let comment_density = (comment_lines as f64 / total_lines as f64) * 100.0;
        
        // Check RDF triplet format: // RDF: <Subject> <Predicate> <Object>
        let rdf_regex = Regex::new(r"RDF:\s*(\S+)\s+(\S+)\s+(\S+)").unwrap();
        let triplet_format_valid = lines.iter()
            .filter(|line| line.contains("RDF:"))
            .all(|line| rdf_regex.is_match(line));

        RDFCompliance {
            has_rdf_comments: rdf_lines > 0,
            comment_density,
            triplet_format_valid,
            min_density_met: comment_density >= self.validation_rules.min_rdf_density,
        }
    }

    /// Analyze persona attribution based on content
    fn analyze_persona_attribution(&self, contents: &str) -> PersonaAttribution {
        let content_lower = contents.to_lowercase();
        let mut best_persona = None;
        let mut best_confidence = 0.0;
        let mut found_keywords = Vec::new();

        for (persona, keywords) in &self.elite_personas {
            let mut matches = 0;
            let mut matched_keywords = Vec::new();

            for keyword in keywords {
                if content_lower.contains(&keyword.to_lowercase()) {
                    matches += 1;
                    matched_keywords.push(keyword.clone());
                }
            }

            let confidence = matches as f64 / keywords.len() as f64;
            if confidence > best_confidence {
                best_confidence = confidence;
                best_persona = Some(persona.clone());
                found_keywords = matched_keywords;
            }
        }

        // Check for explicit persona attribution in comments
        for persona in self.elite_personas.keys() {
            if contents.contains(persona) {
                best_persona = Some(persona.clone());
                best_confidence = 1.0;
                break;
            }
        }

        PersonaAttribution {
            assigned_persona: best_persona.clone(),
            persona_confidence: best_confidence,
            elite_team_member: best_persona.is_some(),
            persona_keywords: found_keywords,
        }
    }

    /// Analyze code quality metrics
    fn analyze_code_quality(&self, contents: &str, file_type: &FileType) -> CodeQuality {
        let lines: Vec<&str> = contents.lines().collect();
        let line_count = lines.len();

        // Function counting (simplified)
        let function_count = match file_type {
            FileType::RustSource => contents.matches("fn ").count(),
            FileType::PythonSource => contents.matches("def ").count(),
            FileType::TypeScriptSource => contents.matches("function ").count() + contents.matches("=> ").count(),
            _ => 0,
        };

        // Simple complexity score (functions per 100 lines)
        let complexity_score = if line_count > 0 {
            (function_count as f64 / line_count as f64) * 100.0
        } else {
            0.0
        };

        // Check for test presence
        let has_tests = contents.contains("#[test]") || 
                       contents.contains("def test_") ||
                       contents.contains("it(") ||
                       contents.contains("describe(");

        // Documentation coverage (simplified - comments ratio)
        let comment_lines = lines.iter()
            .filter(|line| line.trim().starts_with("//") || line.trim().starts_with("/*") || line.trim().starts_with("#"))
            .count();
        
        let documentation_coverage = if line_count > 0 {
            (comment_lines as f64 / line_count as f64) * 100.0
        } else {
            0.0
        };

        CodeQuality {
            line_count,
            function_count,
            complexity_score,
            has_tests,
            documentation_coverage,
        }
    }

    /// Analyze security compliance
    fn analyze_security_compliance(&self, contents: &str) -> SecurityCompliance {
        let mut no_dangerous_patterns = true;
        
        for pattern in &self.validation_rules.dangerous_patterns {
            if contents.contains(pattern) {
                no_dangerous_patterns = false;
                break;
            }
        }

        // Check for hardcoded secrets (simple patterns)
        let secret_patterns = [
            r"password\s*=\s*['\"][^'\"]+['\"]",
            r"api_key\s*=\s*['\"][^'\"]+['\"]",
            r"secret\s*=\s*['\"][^'\"]+['\"]",
            r"token\s*=\s*['\"][a-zA-Z0-9]{20,}['\"]",
        ];

        let no_hardcoded_secrets = !secret_patterns.iter()
            .any(|pattern| Regex::new(pattern).unwrap().is_match(contents));

        // Check for safe crypto usage (presence of crypto imports/usage)
        let crypto_usage_safe = contents.contains("use sha2") || 
                               contents.contains("use crypto") ||
                               contents.contains("hashlib") ||
                               !contents.contains("md5"); // MD5 is considered unsafe

        // Check for input validation patterns
        let input_validation_present = contents.contains("validate") ||
                                      contents.contains("sanitize") ||
                                      contents.contains("escape") ||
                                      contents.contains("Result<") ||
                                      contents.contains("Option<");

        SecurityCompliance {
            no_hardcoded_secrets,
            no_dangerous_patterns,
            crypto_usage_safe,
            input_validation_present,
        }
    }

    /// Generate Dr. Langford's cryptographic signature
    fn generate_langford_signature(&self, file_path: &str, compliance_score: f64) -> String {
        let mut hasher = Sha256::new();
        hasher.update(file_path);
        hasher.update(&compliance_score.to_string());
        hasher.update("Dr. Adelaide V. Langford");
        hasher.update(&Utc::now().to_rfc3339());
        
        let hash = hasher.finalize();
        format!("QK-L1-{:X}-AUTH-HEX", &hash[0..4].iter().fold(0u32, |acc, &b| (acc << 8) | b as u32))
    }

    /// Execute Dr. Langford's validation on a single file
    fn validate_file_with_langford(&self, path: &Path, contents: &str, file_type: &FileType) -> LangfordValidation {
        let mut issues = Vec::new();
        let mut compliance_score = 100.0;

        // USIM compliance check
        let usim = self.validate_usim_compliance(contents);
        if usim.compliance_score < 100.0 {
            compliance_score -= (100.0 - usim.compliance_score) * 0.3;
            issues.push(ValidationIssue {
                category: "USIM Compliance".to_string(),
                severity: "High".to_string(),
                description: "Missing or invalid USIM headers (hashid, cuid, ttl)".to_string(),
                line_number: Some(1),
                remediation: "Add valid USIM headers in file header comment".to_string(),
            });
        }

        // RDF compliance check
        let rdf = self.validate_rdf_compliance(contents);
        if !rdf.min_density_met {
            compliance_score -= 20.0;
            issues.push(ValidationIssue {
                category: "RDF Compliance".to_string(),
                severity: "Medium".to_string(),
                description: format!("RDF comment density {:.1}% below required 30%", rdf.comment_density),
                line_number: None,
                remediation: "Add RDF triplet comments: // RDF: <Subject> <Predicate> <Object>".to_string(),
            });
        }

        // Persona attribution check
        let persona = self.analyze_persona_attribution(contents);
        if !persona.elite_team_member {
            compliance_score -= 15.0;
            issues.push(ValidationIssue {
                category: "Persona Attribution".to_string(),
                severity: "Medium".to_string(),
                description: "No elite team persona assigned to file".to_string(),
                line_number: None,
                remediation: "Assign file to appropriate elite team member".to_string(),
            });
        }

        // Security compliance check
        let security = self.analyze_security_compliance(contents);
        if !security.no_dangerous_patterns || !security.no_hardcoded_secrets {
            compliance_score -= 30.0;
            issues.push(ValidationIssue {
                category: "Security Compliance".to_string(),
                severity: "Critical".to_string(),
                description: "Security vulnerabilities detected".to_string(),
                line_number: None,
                remediation: "Remove dangerous patterns and hardcoded secrets".to_string(),
            });
        }

        let validation_passed = compliance_score >= 80.0;
        let path_str = path.to_string_lossy().to_string();
        let crypto_signature = self.generate_langford_signature(&path_str, compliance_score);

        LangfordValidation {
            inspector: "Dr. Adelaide V. Langford".to_string(),
            validation_passed,
            compliance_score,
            crypto_signature,
            issues,
            disa_clearance: if validation_passed { "APPROVED" } else { "CONDITIONAL" }.to_string(),
        }
    }

    /// Main scanning function with Dr. Langford's enhanced validation
    pub fn scan_repository(&self, repo_path: &Path) -> CTASValidationReport {
        let start_time = std::time::Instant::now();
        let mut files = Vec::new();
        let mut file_count = 0;

        // Recursive directory traversal with ignore pattern support
        let _ = self.visit_dirs_enhanced(repo_path, &mut |entry: &DirEntry| {
            let path = entry.path();
            
            // Skip ignored files
            if self.should_ignore(&path) {
                return;
            }

            let file_type = self.determine_file_type(&path);
            
            // Only process code files and relevant config files
            if !matches!(file_type, FileType::RustSource | FileType::PythonSource | FileType::TypeScriptSource | FileType::JavaScriptSource | FileType::ConfigFile) {
                return;
            }

            file_count += 1;

            let contents = fs::read_to_string(&path).unwrap_or_else(|_| String::new());
            let is_complete = self.is_code_complete(&contents, &file_type);

            // Dr. Langford's comprehensive validation
            let langford_validation = self.validate_file_with_langford(&path, &contents, &file_type);
            let persona_attribution = self.analyze_persona_attribution(&contents);
            let usim_compliance = self.validate_usim_compliance(&contents);
            let rdf_compliance = self.validate_rdf_compliance(&contents);
            let code_quality = self.analyze_code_quality(&contents, &file_type);
            let security_compliance = self.analyze_security_compliance(&contents);

            files.push(EnhancedFileStatus {
                path: path.to_string_lossy().to_string(),
                file_type,
                is_complete,
                langford_validation,
                persona_attribution,
                usim_compliance,
                rdf_compliance,
                code_quality,
                security_compliance,
            });
        });

        let scan_duration = start_time.elapsed();

        // Generate summary statistics
        let summary = self.generate_summary(&files);
        let elite_team_status = self.generate_elite_team_status(&files);
        let langford_verdict = self.generate_langford_overall_verdict(&files);
        let recommendations = self.generate_recommendations(&files);

        CTASValidationReport {
            metadata: ReportMetadata {
                inspector: "Dr. Adelaide V. Langford - QuantaKnife".to_string(),
                scan_timestamp: Utc::now(),
                repo_path: repo_path.to_string_lossy().to_string(),
                total_files_scanned: file_count,
                scan_duration_ms: scan_duration.as_millis(),
                ctas_version: "6.6.0".to_string(),
            },
            summary,
            files,
            langford_verdict,
            elite_team_status,
            recommendations,
        }
    }

    /// Enhanced directory visitor with ignore pattern support
    fn visit_dirs_enhanced(&self, dir: &Path, cb: &mut dyn FnMut(&DirEntry)) -> std::io::Result<()> {
        if dir.is_dir() && !self.should_ignore(dir) {
            for entry in fs::read_dir(dir)? {
                let entry = entry?;
                let path = entry.path();
                if path.is_dir() {
                    self.visit_dirs_enhanced(&path, cb)?;
                } else {
                    cb(&entry);
                }
            }
        }
        Ok(())
    }

    /// Generate validation summary statistics
    fn generate_summary(&self, files: &[EnhancedFileStatus]) -> ValidationSummary {
        let total_code_files = files.len();
        let complete_files = files.iter().filter(|f| f.is_complete).count();
        let incomplete_files = total_code_files - complete_files;
        let persona_assigned_files = files.iter().filter(|f| f.persona_attribution.elite_team_member).count();
        let usim_compliant_files = files.iter().filter(|f| f.usim_compliance.compliance_score >= 100.0).count();
        let rdf_compliant_files = files.iter().filter(|f| f.rdf_compliance.min_density_met).count();

        let overall_compliance_score = if total_code_files > 0 {
            files.iter()
                .map(|f| f.langford_validation.compliance_score)
                .sum::<f64>() / total_code_files as f64
        } else {
            0.0
        };

        ValidationSummary {
            total_code_files,
            complete_files,
            incomplete_files,
            persona_assigned_files,
            usim_compliant_files,
            rdf_compliant_files,
            overall_compliance_score,
        }
    }

    /// Generate elite team status breakdown
    fn generate_elite_team_status(&self, files: &[EnhancedFileStatus]) -> EliteTeamStatus {
        let mut status = EliteTeamStatus {
            natasha_volkov_files: 0,
            omar_al_rashid_files: 0,
            james_sterling_files: 0,
            emily_chen_files: 0,
            dmitri_kozlov_files: 0,
            aria_nakamura_files: 0,
            adelaide_langford_files: 0,
            unassigned_files: 0,
        };

        for file in files {
            match &file.persona_attribution.assigned_persona {
                Some(persona) => {
                    match persona.as_str() {
                        "Natasha Volkov" => status.natasha_volkov_files += 1,
                        "Omar Al-Rashid" => status.omar_al_rashid_files += 1,
                        "James Sterling" => status.james_sterling_files += 1,
                        "Emily Chen" => status.emily_chen_files += 1,
                        "Dmitri Kozlov" => status.dmitri_kozlov_files += 1,
                        "Aria Nakamura" => status.aria_nakamura_files += 1,
                        "Dr. Adelaide V. Langford" => status.adelaide_langford_files += 1,
                        _ => status.unassigned_files += 1,
                    }
                },
                None => status.unassigned_files += 1,
            }
        }

        status
    }

    /// Generate Dr. Langford's overall verdict
    fn generate_langford_overall_verdict(&self, files: &[EnhancedFileStatus]) -> LangfordOverallVerdict {
        let total_files = files.len();
        let critical_issues: Vec<String> = files.iter()
            .flat_map(|f| &f.langford_validation.issues)
            .filter(|issue| issue.severity == "Critical")
            .map(|issue| issue.description.clone())
            .collect();

        let overall_score = if total_files > 0 {
            files.iter()
                .map(|f| f.langford_validation.compliance_score)
                .sum::<f64>() / total_files as f64
        } else {
            0.0
        };

        let final_result = if critical_issues.len() > 0 {
            "CRITICAL_FAILURE"
        } else if overall_score >= 90.0 {
            "APPROVED"
        } else if overall_score >= 70.0 {
            "CONDITIONAL"
        } else {
            "REJECTED"
        };

        let authority_statement = match final_result {
            "APPROVED" => "Dr. Langford certifies system compliance with DISA standards for operational deployment",
            "CONDITIONAL" => "Dr. Langford requires improvements before full operational certification",
            "REJECTED" => "Dr. Langford requires significant remediation before operational consideration",
            "CRITICAL_FAILURE" => "Dr. Langford demands immediate halt and remediation of critical vulnerabilities",
            _ => "Dr. Langford validation status unknown",
        };

        LangfordOverallVerdict {
            final_result: final_result.to_string(),
            overall_score,
            crypto_signature: self.generate_langford_signature("OVERALL_VERDICT", overall_score),
            authority_statement: authority_statement.to_string(),
            critical_issues,
        }
    }

    /// Generate actionable recommendations
    fn generate_recommendations(&self, files: &[EnhancedFileStatus]) -> Vec<String> {
        let mut recommendations = Vec::new();

        let incomplete_files = files.iter().filter(|f| !f.is_complete).count();
        if incomplete_files > 0 {
            recommendations.push(format!("Complete {} incomplete code files", incomplete_files));
        }

        let unassigned_files = files.iter().filter(|f| !f.persona_attribution.elite_team_member).count();
        if unassigned_files > 0 {
            recommendations.push(format!("Assign {} files to elite team personas", unassigned_files));
        }

        let usim_non_compliant = files.iter().filter(|f| f.usim_compliance.compliance_score < 100.0).count();
        if usim_non_compliant > 0 {
            recommendations.push(format!("Add USIM headers to {} files", usim_non_compliant));
        }

        let rdf_non_compliant = files.iter().filter(|f| !f.rdf_compliance.min_density_met).count();
        if rdf_non_compliant > 0 {
            recommendations.push(format!("Improve RDF comment density in {} files", rdf_non_compliant));
        }

        let security_issues = files.iter().filter(|f| !f.security_compliance.no_dangerous_patterns || !f.security_compliance.no_hardcoded_secrets).count();
        if security_issues > 0 {
            recommendations.push(format!("Address security issues in {} files", security_issues));
        }

        if recommendations.is_empty() {
            recommendations.push("System meets Dr. Langford's validation standards".to_string());
        }

        recommendations
    }
}

/// Main function - CLI interface for enhanced scanner
fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔐 CTAS Enhanced Monorepo Scanner - Dr. Langford Integration");
    println!("=============================================================");
    println!("DISA External Validation Authority - 'QuantaKnife' Standards");
    println!("\"Precision is mercy. Incomplete logic is a liability.\"\n");

    let repo_root = Path::new(".");
    let scanner = CTASEnhancedScanner::new();
    
    println!("🔍 Scanning repository with Dr. Langford's validation criteria...");
    let report = scanner.scan_repository(repo_root);

    // Output console summary
    println!("\n📊 CTAS VALIDATION SUMMARY");
    println!("========================");
    println!("Inspector: {}", report.metadata.inspector);
    println!("Scan Duration: {}ms", report.metadata.scan_duration_ms);
    println!("Total Files: {}", report.summary.total_code_files);
    println!("Complete Files: {}", report.summary.complete_files);
    println!("Incomplete Files: {}", report.summary.incomplete_files);
    println!("Persona Assigned: {}", report.summary.persona_assigned_files);
    println!("USIM Compliant: {}", report.summary.usim_compliant_files);
    println!("RDF Compliant: {}", report.summary.rdf_compliant_files);
    println!("Overall Compliance: {:.1}%", report.summary.overall_compliance_score);

    // Dr. Langford's Verdict
    println!("\n🔐 DR. LANGFORD'S VERDICT");
    println!("========================");
    println!("Result: {}", report.langford_verdict.final_result);
    println!("Score: {:.1}%", report.langford_verdict.overall_score);
    println!("Signature: {}", report.langford_verdict.crypto_signature);
    println!("Authority: {}", report.langford_verdict.authority_statement);

    if !report.langford_verdict.critical_issues.is_empty() {
        println!("\n🚨 CRITICAL ISSUES:");
        for issue in &report.langford_verdict.critical_issues {
            println!("  • {}", issue);
        }
    }

    // Elite Team Status
    println!("\n👥 ELITE TEAM FILE ATTRIBUTION");
    println!("==============================");
    println!("🇷🇺 Natasha Volkov: {}", report.elite_team_status.natasha_volkov_files);
    println!("🇲🇦 Omar Al-Rashid: {}", report.elite_team_status.omar_al_rashid_files);
    println!("🇺🇸 James Sterling: {}", report.elite_team_status.james_sterling_files);
    println!("🇺🇸 Emily Chen: {}", report.elite_team_status.emily_chen_files);
    println!("🇷🇺 Dmitri Kozlov: {}", report.elite_team_status.dmitri_kozlov_files);
    println!("🇯🇵 Aria Nakamura: {}", report.elite_team_status.aria_nakamura_files);
    println!("🔐 Dr. Langford: {}", report.elite_team_status.adelaide_langford_files);
    println!("❓ Unassigned: {}", report.elite_team_status.unassigned_files);

    // Recommendations
    println!("\n💡 DR. LANGFORD'S RECOMMENDATIONS");
    println!("================================");
    for (i, rec) in report.recommendations.iter().enumerate() {
        println!("{}. {}", i + 1, rec);
    }

    // Save detailed JSON report
    let json_report = serde_json::to_string_pretty(&report)?;
    let report_filename = format!("ctas_langford_validation_report_{}.json", 
        report.metadata.scan_timestamp.format("%Y%m%d_%H%M%S"));
    
    fs::write(&report_filename, json_report)?;
    println!("\n📋 Detailed report saved: {}", report_filename);

    // Final authority statement
    match report.langford_verdict.final_result.as_str() {
        "APPROVED" => println!("\n✅ DISA AUTHORITY: System approved for operational deployment"),
        "CONDITIONAL" => println!("\n⚠️  DISA AUTHORITY: Conditional approval - address recommendations"),
        "REJECTED" => println!("\n❌ DISA AUTHORITY: System rejected - significant remediation required"),
        "CRITICAL_FAILURE" => println!("\n🚨 DISA AUTHORITY: CRITICAL FAILURE - HALT ALL OPERATIONS"),
        _ => {},
    }

    println!("\n🔐 Cryptographic Authority: Dr. Adelaide V. Langford - 'QuantaKnife'");
    println!("DISA External Validation Lead - OSI-Layer Forensics");

    Ok(())
}