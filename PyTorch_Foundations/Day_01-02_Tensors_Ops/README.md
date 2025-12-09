# 🧠 Day 01 & 02: Tensors, Broadcasting & Ops

**Status:** ✅ Complete | **Curriculum:** Level 1 Foundations

This folder covers the first **two days** of the 10-day curriculum.
I merged these topics because they form a single, inseparable unit: **The Geometry of Deep Learning.** Before building neural networks, I had to master the raw mechanics of data manipulation—specifically, how to perform complex mathematical operations without using slow Python loops.

---

## 🎯 Concepts Mastered

### Day 01: Tensors & Ops

- **The "Shape is King" Rule:** In Deep Learning, values don't matter if the geometry is wrong.
- **Reshaping (`.view`):** Rearranging memory without changing data (Conservation of Mass).
- **The Batch Dimension:** Understanding that models expect a "tray" of data `(Batch, Channels, Height, Width)`, not just a single item.

### Day 02: Broadcasting & Vectorization

- **Broadcasting:** The mathematical magic that stretches dimensions of size `1` to match larger dimensions.
- **Vectorization:** Replacing loops with matrix operations (`@`) for massive speedups.
- **Matrix Multiplication:** Understanding the strict alignment rules ($M \times N$ @ $N \times P$).

---

## 🛠️ The Code (Proof of Work)

I built three focused implementations to solve specific engineering problems using **zero-loop** logic.

| Notebook                                                                 | Topic Covered     | Engineering Problem Solved                                         |
| :----------------------------------------------------------------------- | :---------------- | :----------------------------------------------------------------- |
| [**01_normalization_pipeline.ipynb**](./01_normalization_pipeline.ipynb) | **Broadcasting**  | A production-ready image normalization pipeline in 2 lines.        |
| [**02_slicing_and_mm.ipynb**](./02_slicing_and_mm.ipynb)                 | **Vectorization** | Linear transformations using Matrix Multiplication (`@`).          |
| [**03_reduction_and_dtype.ipynb**](./03_reduction_and_dtype.ipynb)       | **Ops Stability** | Calculating stats (`mean`) across dimensions without type crashes. |

---

## 💡 "Aha!" Moments

### 1. The Dtype Crash (Day 01 Insight)

I learned that PyTorch is strict about types.

- **Error:** `RuntimeError` when running `.mean()` on Integers (`Long`).
- **Fix:** Always cast to float first: `tensor.float().mean()`.

### 2. The Commutative Trap (Day 02 Insight)

Matrix multiplication is **not** commutative.

- Shape `(2, 3)` @ `(3, 2)` $\to$ `(2, 2)`
- Shape `(3, 2)` @ `(2, 3)` $\to$ `(3, 3)`
- _Lesson:_ The order of operands strictly dictates the output geometry.

---

## 🚀 How to Run

1.  Open the `.ipynb` files in [Google Colab](https://colab.research.google.com/) or Jupyter Lab.
2.  Run all cells to see the shape verifications.

_Next Stop: Day 03 - Autograd_
