## Multilayer Perceptron

```
Input(N) → Linear(128) → ReLU → BatchNorm1d  
          → Linear(128) → ReLU → BatchNorm1d  
          → Linear(64) → ReLU  
          → Linear(1) → Sigmoid
```