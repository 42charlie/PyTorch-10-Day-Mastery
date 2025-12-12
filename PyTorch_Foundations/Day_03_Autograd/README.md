# 🧠 Day 03: Autograd & The Computational Graph

**Status:** ✅ Complete | **Curriculum:** Level 1 Foundations

This folder covers **Day 03** of the curriculum.
After mastering the geometry of tensors (Days 1 & 2), I moved to the "engine" that makes training possible: **Automatic Differentiation**. I learned that I don't need to be a calculus wizard to build models—I just need to understand how PyTorch records history to calculate gradients.

---

## 🎯 Concepts Mastered

### 1. The Dynamic Computational Graph (DAG)

- **The Concept:** PyTorch builds a graph on the fly. Tensors are nodes, and operations (like `+` or `*`) are the edges.
- **Forward Pass:** Running the code normally to compute the output (Prediction).
- **Backward Pass:** Traversing the graph in reverse to calculate derivatives (Gradients).

### 2. The Mechanics of `.backward()`

- **The Trigger:** Calling this method on the `Loss` kicks off the chain reaction.
- **The Chain Rule:** The mathematical logic PyTorch uses: $\text{Gradient} = \text{Outer Derivative} \times \text{Inner Derivative}$.
- **Tracking:** The vital importance of `requires_grad=True` for model parameters (Weights/Biases) vs. `False` for static data.

---

## 🛠️ The Code (Proof of Work)

I focused on a single, rigorous verification exercise to prove that PyTorch's "magic" is actually just precise math.

| Notebook                                                                             | Topic Covered                | Engineering Problem Solved                                                                                    |
| :----------------------------------------------------------------------------------- | :--------------------------- | :------------------------------------------------------------------------------------------------------------ |
| [**Day_03_manual_neuron_verification.ipynb**](./Day_03_manual_neuron_verification.ipynb) | **Autograd vs. Manual Math** | Building a "Manual Neuron" to mathematically prove that `.backward()` yields the correct partial derivatives. |

---

## ⚡ Key Insights

### 1. The "Onion" Visualization (Chain Rule)

I realized that calculating gradients is like peeling an onion.

- **Outer Layer:** How the Loss changes when the Prediction changes.
- **Inner Layer:** How the Prediction changes when the Weight changes.
- **Insight:** PyTorch simply multiplies these "layers" together.

### 2. The "Bias" Shortcut

I discovered why Bias terms are simple.

- **Math:** $\frac{\partial}{\partial b}(w \cdot x + b) = 1$.
- **Insight:** Because the addition operation doesn't scale the input, the bias node acts as a "gradient distributor," passing the incoming signal through unchanged (multiplying by 1).

### 3. Frozen Variables

I understood why we treat variables as constants during partial differentiation.

- When calculating `d(Loss)/db`, the weight `w` and input `x` are frozen. Their derivative is 0 relative to `b`. This simplifies the mental model of backpropagation significantly.

---

## 🚀 How to Run

1. Open the `.ipynb` file in [Google Colab](https://colab.research.google.com/) or Jupyter Lab.
2. Run the cells to see the comparison: `PyTorch Gradient` vs `My Manual Calculation`.

_Next Stop: Day 04 - nn.Module Basics_
