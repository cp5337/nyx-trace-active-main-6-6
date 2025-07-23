# NyxTrace Dioxus Migration Guide

## Overview

This guide outlines the approach for migrating the NyxTrace/CTAS system from Streamlit to Dioxus, a Rust-based reactive framework for building cross-platform user interfaces. This migration aligns with the broader goal of transitioning the codebase to Rust for improved performance and reliability.

## Why Dioxus?

Dioxus offers several advantages for the NyxTrace system:

1. **Rust-based**: Aligns with the project's goal of migrating to Rust
2. **Cross-platform**: Can target web, desktop, mobile, and TUI
3. **React-like API**: Familiar component model and hooks pattern
4. **Performance**: Significantly better performance than Python-based frameworks
5. **Type safety**: Rust's strong type system prevents many common errors
6. **Memory safety**: Rust's ownership model prevents memory leaks and race conditions

## Component Mapping

### Streamlit → Dioxus Component Mapping

| Streamlit Component | Dioxus Equivalent | Notes |
|---------------------|-------------------|-------|
| `st.container()` | `cx.render(rsx! { div { ... } })` | Use standard HTML elements or custom components |
| `st.columns([1,1])` | `rsx! { div { class: "grid", ... } }` | Use CSS Grid or Flexbox for layout |
| `st.expander()` | Custom `Expander` component | Implement collapsible sections |
| `st.tabs()` | Custom `Tabs` component | Implement tabbed interface |
| `st.sidebar` | Custom `Sidebar` component | Create a persistent sidebar layout |
| `st.markdown()` | `dioxus-markdown` crate | Use the Markdown rendering component |
| `st.write()` | `rsx! { p { ... } }` | Use standard HTML elements |
| `st.metric()` | Custom `Metric` component | Create a reusable metric card component |
| `st.dataframe()` | Custom `DataTable` component | Implement a data grid component |
| `st.plotly_chart()` | `dioxus-charts` or WebAssembly | Use Rust charting libraries or WebAssembly |

### Project Structure

```
src/
├── components/       # Reusable UI components
│   ├── common/       # Shared components like buttons, cards
│   ├── layout/       # Layout components like sidebar, containers
│   ├── data/         # Data visualization components
│   └── specialized/  # Domain-specific components
├── pages/            # Page components
├── hooks/            # Custom hooks for state and effects
├── api/              # API integration
├── state/            # Application state management
├── models/           # Data models and structures
└── utils/            # Utility functions
```

## State Management

### Approach

1. Use Dioxus's built-in hooks for component state:
   ```rust
   let (count, set_count) = use_state(cx, || 0);
   ```

2. For global state, use Dioxus's `use_shared_state`:
   ```rust
   use_shared_state_provider(cx, || AppState::default());
   let app_state = use_shared_state::<AppState>(cx).unwrap();
   ```

3. For complex state management, consider using the Fermi crate:
   ```rust
   use fermi::*;
   
   static TASKS: AtomRef<Vec<Task>> = |_| vec![];
   
   fn TaskList(cx: Scope) -> Element {
       let tasks = use_atom_ref(cx, TASKS);
       // ...
   }
   ```

### Session State Migration

Convert Streamlit's session state to Dioxus state management:

```python
# Streamlit
if 'selected_tasks' not in st.session_state:
    st.session_state.selected_tasks = []
```

```rust
// Dioxus
let (selected_tasks, set_selected_tasks) = use_state(cx, Vec::<Task>::new);
```

## API Integration

### Backend API

1. Create a Rust backend using Axum or Actix Web
2. Implement RESTful API endpoints for all data access
3. Use the `reqwest` crate for HTTP requests from the frontend
4. Serialize/deserialize data using Serde

### Example API Integration

```rust
use reqwest::Client;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
struct Task {
    id: String,
    hash_id: String,
    task_name: String,
    // ... other fields
}

async fn fetch_tasks() -> Result<Vec<Task>, reqwest::Error> {
    let client = Client::new();
    let response = client.get("https://api.example.com/tasks")
        .send()
        .await?
        .json::<Vec<Task>>()
        .await?;
    
    Ok(response)
}

fn TaskList(cx: Scope) -> Element {
    let future = use_future(cx, (), |_| async move {
        fetch_tasks().await
    });

    match future.value() {
        Some(Ok(tasks)) => {
            cx.render(rsx! {
                div {
                    tasks.iter().map(|task| {
                        rsx! { TaskCard { task: task.clone() } }
                    })
                }
            })
        }
        Some(Err(e)) => {
            cx.render(rsx! {
                div { "Error loading tasks: {e}" }
            })
        }
        None => {
            cx.render(rsx! {
                div { "Loading tasks..." }
            })
        }
    }
}
```

## Task Card Implementation

The Task Card component is a critical part of the NyxTrace interface:

```rust
#[derive(Props, PartialEq)]
struct TaskCardProps {
    task: Task,
    compact: Option<bool>,
}

fn TaskCard(cx: Scope<TaskCardProps>) -> Element {
    let task = &cx.props.task;
    let compact = cx.props.compact.unwrap_or(true);
    
    let reliability = task.reliability.unwrap_or(0.5);
    let confidence = task.confidence.unwrap_or(0.5);
    
    let phase_class = format!("phase-{}", task.phase.to_lowercase());
    
    cx.render(rsx! {
        div {
            class: "task-card",
            style: "border-color: {task.color};",
            
            div {
                class: "task-category-label",
                style: "background-color: {task.color}80;",
                "{task.category}"
            }
            
            div {
                class: "task-card-header",
                
                div {
                    class: "task-symbol-container",
                    
                    div {
                        class: "task-symbol",
                        style: "background-color: {task.color};",
                        "{task.symbol}"
                    }
                    
                    div {
                        class: "task-id-container",
                        div {
                            class: "task-hash-id",
                            "{task.hash_id}"
                        }
                        div {
                            class: "task-id",
                            "UUID: {task.id.split('-').nth(1).unwrap_or('N/A')}"
                        }
                    }
                }
            }
            
            div {
                class: "task-name",
                "🧠 Persona: \"{task.category.chars().take(4).collect::<String>()}\""
            }
            
            // Render additional details if not compact
            if !compact {
                rsx! {
                    div {
                        class: "task-description",
                        "{task.description}"
                    }
                    
                    MetricBar {
                        label: "Reliability".to_string(),
                        value: reliability,
                        color: "#3B82F6".to_string()
                    }
                    
                    MetricBar {
                        label: "Confidence".to_string(),
                        value: confidence,
                        color: "#10B981".to_string()
                    }
                    
                    // Additional details...
                }
            }
        }
    })
}
```

## Styling Approach

Dioxus supports several styling approaches:

1. **CSS Files**: Import traditional CSS files
2. **CSS-in-Rust**: Use the `dioxus-free-style` crate
3. **Tailwind CSS**: Use the `dioxus-tailwind` crate

Example with CSS-in-Rust:

```rust
use dioxus_free_style::*;

// Create styles
let task_card = css!("
    border: 2px solid #4A4A4A;
    border-radius: 8px;
    padding: 12px;
    background-color: #1E293B;
    color: white;
    margin-bottom: 10px;
    position: relative;
    overflow: hidden;
    transition: all 0.3s ease;
    cursor: pointer;
");

// Apply in component
cx.render(rsx! {
    div {
        class: "{task_card}",
        // Component content
    }
})
```

## Testing Strategy

1. Unit tests for components using `dioxus-test`:
   ```rust
   #[test]
   fn test_task_card() {
       let mut test = Harness::new(|cx| {
           let task = Task {
               id: "123".to_string(),
               hash_id: "TEST-001".to_string(),
               task_name: "Test Task".to_string(),
               // ...
           };
           
           rsx! { cx, TaskCard { task: task } }
       });
       
       assert!(test.contains("TEST-001"));
   }
   ```

2. Integration tests for page components
3. End-to-end tests with WebDriver or similar

## Incremental Migration Strategy

1. Start with Core Components
   - Begin by implementing the base UI components
   - Focus on Task Card, Navigation, and basic layouts

2. Create Page Framework
   - Implement the main application shell
   - Set up routing using `dioxus-router`

3. Implement Data Fetching
   - Create API integration layer
   - Implement state management

4. Migrate Pages One by One
   - Start with simpler pages
   - Move to more complex interactive pages
   - Run Streamlit and Dioxus in parallel during transition

5. Integration and Testing
   - Ensure feature parity with the Streamlit version
   - Perform comprehensive testing

## Development Environment Setup

```toml
# Cargo.toml
[dependencies]
dioxus = "0.4.0"
dioxus-web = "0.4.0"
dioxus-router = "0.4.0"
dioxus-free-style = "0.4.0"
dioxus-markdown = "0.4.0"
fermi = "0.4.0"
reqwest = { version = "0.11", features = ["json"] }
serde = { version = "1.0", features = ["derive"] }
serde_json = "1.0"
```

## Building and Running

For web targets:
```bash
cargo install dioxus-cli
dx serve
```

For desktop targets:
```bash
dx build --desktop
```

## Additional Resources

- [Dioxus Documentation](https://dioxuslabs.com/docs/0.4/)
- [Dioxus Examples](https://github.com/DioxusLabs/example-projects)
- [Fermi State Management](https://github.com/DioxusLabs/fermi)
- [Dioxus Router](https://github.com/DioxusLabs/dioxus/tree/main/packages/router)