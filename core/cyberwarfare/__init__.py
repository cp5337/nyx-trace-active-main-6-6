"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-CORE-CYBERWARFARE-INIT-0001         │
// │ 📁 domain       : Cyberwarfare, Offensive Security         │
// │ 🧠 description  : Cyberwarfare tools package               │
// │                  Offensive security tool integration       │
// │ 🕸️ hash_type    : UUID → CUID-linked module                │
// │ 🔄 parent_node  : NODE_CORE                                │
// │ 🧩 dependencies : subprocess, pandas, streamlit             │
// │ 🔧 tool_usage   : Offensive Analysis                       │
// │ 📡 input_type   : Tool configurations, commands             │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : cyberwarfare automation, integration     │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Cyberwarfare Core Package
----------------------
This package provides offensive security tool integration, automation, and analysis
capabilities, focusing on cyberwarfare operations, vulnerability scanning,
penetration testing, and threat intelligence gathering.

Designed for future Rust compatibility with clear interfaces and types.
"""

from .kali_integrator import (
    KaliIntegrator,
    ToolCategory,
    execute_command,
    parse_command_output,
    get_available_tools,
    check_tool_installed,
    CommandResult,
)

from .tool_manager import (
    ToolManager,
    CyberTool,
    CommandTemplate,
    get_tool_categories,
    get_tools_by_category,
    format_command_template,
)

from .tool_scraper import ToolScraper, ToolInfo, ScraperResult

from .assessment_reporting import (
    CyberReport,
    create_vulnerability_report,
    export_report,
    generate_report_summary,
)

from .results_parser import (
    parse_nmap_results,
    parse_dirb_results,
    parse_nikto_results,
    parse_gobuster_results,
    parse_whatweb_results,
)
