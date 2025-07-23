"""
// ┌─────────────────────────────────────────────────────────────┐
// │ █████████████████ CTAS USIM HEADER ███████████████████████ │
// ├─────────────────────────────────────────────────────────────┤
// │ 🔖 hash_id      : USIM-CORE-SECURITY-INIT-0001             │
// │ 📁 domain       : Security, Cybersecurity                  │
// │ 🧠 description  : Security core package                     │
// │                  Cybersecurity tool integration            │
// │ 🕸️ hash_type    : UUID → CUID-linked module                │
// │ 🔄 parent_node  : NODE_CORE                                │
// │ 🧩 dependencies : subprocess, pandas, streamlit             │
// │ 🔧 tool_usage   : Security Analysis                        │
// │ 📡 input_type   : Tool configurations, commands             │
// │ 🧪 test_status  : stable                                   │
// │ 🧠 cognitive_fn : security automation, integration         │
// │ ⌛ TTL Policy   : 6.5 Persistent                           │
// └─────────────────────────────────────────────────────────────┘

Security Core Package
------------------
This package provides security tool integration, automation, and analysis
capabilities, focusing on cybersecurity operations, vulnerability scanning,
penetration testing, and threat intelligence gathering.
"""

from .kali_integrator import (
    KaliIntegrator,
    ToolCategory,
    execute_command,
    parse_command_output,
    get_available_tools,
    check_tool_installed,
)

from .tool_manager import (
    ToolManager,
    SecurityTool,
    CommandTemplate,
    get_tool_categories,
    get_tools_by_category,
    format_command_template,
)

from .reporting import (
    SecurityReport,
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
