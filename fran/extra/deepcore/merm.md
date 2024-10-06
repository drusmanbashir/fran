# Diagrams
## Mermaid Diagram


```mermaid

graph TD
    A[finish_run] --> B{Model is DataParallel?}
    B -- Yes --> C[Extract Model]
    B -- No --> D[Continue]
    
    D --> E[Disable Grad Calculation]
    
    E --> F{Balance=True?}
    F -- Yes --> G[Select by Class]
    G --> H[Calc Gradients for Class]
    H --> I[Use FacilityLocation Function]
    I --> J[Submodular Optimizer Selects]
    J --> K[Calc Weights]
    
    F -- No --> L[Select Across Dataset]
    L --> M[Calc Gradients for All Classes]
    M --> N[Use FacilityLocation Function]
    N --> O[Submodular Optimizer Selects]
    O --> P[Calc Weights]
    
    P --> Q[Enable Grad Calculation]
    Q --> R[Return Selected Indices and Weights]
 ```
