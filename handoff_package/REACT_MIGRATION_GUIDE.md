# NyxTrace React Migration Guide

## Overview

This guide outlines the approach for migrating the NyxTrace/CTAS system from Streamlit to React. The migration will be incremental, focusing on component-by-component conversion while maintaining functionality.

## Component Mapping

### Streamlit → React Component Mapping

| Streamlit Component | React Equivalent | Notes |
|---------------------|------------------|-------|
| `st.container()` | `<div>` or custom container | Consider using styled-components for styling |
| `st.columns([1,1])` | CSS Grid or Flexbox | Use responsive grid system like MUI Grid |
| `st.expander()` | Accordion component | MUI Accordion or custom implementation |
| `st.tabs()` | Tab component | MUI Tabs or React-Tabs library |
| `st.sidebar` | Drawer or sidebar component | Consider collapsible navigation pattern |
| `st.markdown()` | Markdown rendering library | React-Markdown with HTML sanitization |
| `st.write()` | `<p>` or text component | Use typography system for consistency |
| `st.metric()` | Custom metric component | Create reusable stat/metric cards |
| `st.dataframe()` | Data grid component | Consider React-Table or MUI DataGrid |
| `st.plotly_chart()` | Plotly.js React | Direct integration with Plotly.js |

### Page Structure

Streamlit pages will be converted to React components with routing:

```jsx
// Example route structure
<Routes>
  <Route path="/" element={<Dashboard />} />
  <Route path="/adversary-task-viewer" element={<AdversaryTaskViewer />} />
  <Route path="/periodic-table" element={<PeriodicTable />} />
  <Route path="/relationship-network" element={<RelationshipNetwork />} />
</Routes>
```

## State Management

### Approach

1. Use React Context for global application state
2. Redux or Redux Toolkit for complex state management
3. React Query for server state and data fetching
4. Local component state for UI-specific state

### Session State Migration

Convert Streamlit's session state to React's state management:

```python
# Streamlit
if 'selected_tasks' not in st.session_state:
    st.session_state.selected_tasks = []
```

```jsx
// React with Context
const [selectedTasks, setSelectedTasks] = useState([]);
```

## API Integration

### Backend API

1. Create RESTful API endpoints for all data access
2. Use FastAPI or Flask for the backend API
3. Implement proper authentication and authorization
4. Add API versioning for future compatibility

### Data Fetching

1. Use React Query for data fetching and caching
2. Implement loading states and error handling
3. Add pagination for large datasets
4. Consider implementing GraphQL for complex data requirements

## Component Design

### Guidelines

1. Create atomic components following the Atomic Design methodology
2. Use TypeScript for strong typing and better developer experience
3. Implement proper prop validation and default props
4. Add comprehensive component documentation
5. Create Storybook stories for component showcasing

### Example Structure

```
src/
├── components/
│   ├── atoms/ - Basic building blocks
│   ├── molecules/ - Combinations of atoms
│   ├── organisms/ - Complex UI sections
│   └── templates/ - Page layouts
├── pages/ - Route components
├── hooks/ - Custom React hooks
├── services/ - API integration
├── store/ - State management
└── utils/ - Utility functions
```

## Task Card Implementation

The Task Card component conversion is a critical aspect:

```jsx
// Task Card React Component
const TaskCard = ({ task, compact = true }) => {
  const { 
    hash_id, 
    task_name, 
    description, 
    color, 
    symbol, 
    category 
  } = task;
  
  // Calculate metrics
  const reliability = task.reliability || 0.5;
  const confidence = task.confidence || 0.5;
  
  return (
    <Card 
      sx={{ 
        borderColor: color, 
        borderWidth: 2,
        borderStyle: 'solid',
        position: 'relative',
        padding: 2
      }}
    >
      <Box 
        sx={{ 
          position: 'absolute', 
          right: 0, 
          top: 0, 
          bgcolor: `${color}80`,
          padding: '2px 8px',
          borderBottomLeftRadius: 4
        }}
      >
        {category}
      </Box>
      
      <Box display="flex" alignItems="center" gap={1}>
        <Box 
          sx={{ 
            bgcolor: color, 
            width: 40, 
            height: 40, 
            display: 'flex', 
            alignItems: 'center', 
            justifyContent: 'center',
            borderRadius: 1
          }}
        >
          {symbol}
        </Box>
        <Box>
          <Typography variant="subtitle1" fontFamily="monospace" fontWeight="bold">
            {hash_id}
          </Typography>
          <Typography variant="caption" color="text.secondary">
            {task_name}
          </Typography>
        </Box>
      </Box>
      
      {/* More content based on compact mode */}
      {!compact && (
        <>
          <Typography variant="body2" sx={{ mt: 1 }}>
            {description}
          </Typography>
          <Box sx={{ mt: 2 }}>
            <MetricBar label="Reliability" value={reliability} color="#3B82F6" />
            <MetricBar label="Confidence" value={confidence} color="#10B981" />
          </Box>
        </>
      )}
    </Card>
  );
};
```

## Testing Strategy

1. Unit tests for all React components using React Testing Library
2. Integration tests for component interactions
3. End-to-end tests for critical user flows
4. Visual regression testing for UI components

## Rollout Strategy

1. Begin with shared components library
2. Convert individual pages one at a time
3. Run Streamlit and React in parallel during transition
4. Implement feature parity validation
5. Gradual user migration to React interface