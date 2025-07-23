# UI Screenshots Reference Guide

This directory contains screenshots of the current NyxTrace/CTAS user interface. These screenshots serve as a visual reference for understanding the existing implementation and ensuring accurate recreation during refactoring or migration to React.

## Screenshot Catalog

### Main Dashboard
- `01_ctas_dashboard_control_panel_main_landing_page.png`: The main landing page showing the control panel layout and primary navigation.

### Task Viewer Components
- `ctas_node_card.png`: Reference implementation of the node card design showing proper styling and content organization.

### Interface Elements
- Various UI elements and component screenshots demonstrating the current styling, layout, and interactions.

## Design Guidelines

When implementing these interfaces, consider:

1. **Color Scheme**: The application uses a dark-themed interface with accent colors for categorization and status indicators.

2. **Card Components**: Task cards have a consistent structure with:
   - Color-coded borders based on category
   - Category labels in the top-right corner
   - Phase indicators in the top-left (when applicable)
   - Metric visualizations using horizontal bars

3. **Layout Structure**:
   - Left sidebar for navigation and filters
   - Main content area with responsive grid layouts
   - Modal dialogs for detailed information

4. **Typography**:
   - Monospace fonts for identifiers and codes
   - Sans-serif fonts for general text
   - Consistent heading hierarchy

## React Component Considerations

When converting these interfaces to React:

1. **Component Hierarchy**:
   - Create atomic components for cards, metrics, and indicators
   - Compose these into larger page-level components
   - Ensure consistent prop interfaces

2. **Responsive Design**:
   - Ensure layouts adapt to different screen sizes
   - Consider mobile-first approach for new implementations
   - Use CSS Grid or Flexbox for responsive layouts

3. **State Management**:
   - Identify UI state that needs management (selected items, filters, etc.)
   - Plan for efficient updates and renders
   - Consider component-specific vs. global state

4. **Accessibility**:
   - Maintain proper contrast ratios
   - Include aria attributes
   - Ensure keyboard navigation support

Use these screenshots as references for visual styling and functional requirements, not just as direct templates to copy. The goal is to capture the essence and functionality while potentially improving the implementation during the React migration.