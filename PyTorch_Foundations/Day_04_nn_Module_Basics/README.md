# 🏗️ Day 04: nn.Module - Building the actual frame

**Status:** ✅ Done | **Progress:** Day 4/10

Today was basically the day I stopped doing manual matrix math and started using PyTorch's `nn.Module`. It's way better than tracking weights by hand, but there were some stupid traps I fell into with the model structure.

---

## 🎯 What I actually figured out

### The "Handshake" (Don't break the chain)
Everything in PyTorch is about shapes matching. If Layer 1 spits out 100 values, Layer 2 **has** to be ready to catch 100. If you mess up one number in the `nn.Linear(in, out)`, the whole thing just blows up with a Shape Mismatch error.

### The Batch Reality
I finally got that we don't just feed the model one thing at a time. We give it a "batch" (a stack).
- **Shape like `(8, 20)`** = 8 separate items, each with 20 features.
- The model processes all 8 at the same time, which is why it's so fast.

---

## 🛠️ Proof of Work: The "Signal Compressor"

I built a model that takes a big input (20 features) and squeezes it down through hidden layers until it gives one final answer.

| Notebook | What's inside |
| :--- | :--- |
| [**signal_compressor.ipynb**](./signal_compressor.ipynb) | My custom `SignalCompressor` class with 3 layers and ReLU. |

---

## 💡 Some Lessons Learned

### 1. The Activation Trap (The biggest fail)
I thought just writing `self.relu(y)` inside the forward pass was enough. **Nope.** PyTorch doesn't change `y` unless you tell it to. I had to write `y = self.relu(y)` to actually save the "activated" values. Without that `=` sign, the ReLU does nothing and the model stays linear (aka stupid).

### 2. Layers vs Functions
I was confused why some things go in `__init__` and some don't.
- **Layers with weights** (like `nn.Linear`): Must go in `__init__` so PyTorch can "see" the weights to train them.
- **Pure Math** (like ReLU): Can just be a function in the forward pass because there's nothing to learn there.

### 3. The `super()` thing
You have to call `super().__init__()` at the start of your class. If you forget this, your model is just a regular Python class and won't have any of the PyTorch superpowers (like `.to(device)` or tracking parameters).

---
