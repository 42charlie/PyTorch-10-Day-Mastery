# 📉 Day 05: Forward/Backward + Loss - The Learning Bridge

**Status:** ✅ Done | **Progress:** Day 5/10

Today was the day the model actually started "caring" about being wrong. I moved past just building the structure and started implementing the feedback loop that allows a neural network to learn from its mistakes.

---

## 🎯 What I actually figured out

### The Criterion (The Judge)
The Loss Function (or Criterion) is what tells the model how far off it is. 
- **`nn.MSELoss`**: Great for regression (predicting continuous numbers).
- **`nn.CrossEntropyLoss`**: The go-to for classification. 
One big thing: `CrossEntropyLoss` expects raw **logits** (raw scores). If you put a Softmax at the end of your model and then use this loss, you're essentially doing the math twice, which messes up the gradients.

### The `.backward()` Magic
When you call `loss.backward()`, PyTorch walks through the **Computational Graph** in reverse. It uses the Chain Rule to calculate exactly how much each weight contributed to the total error. These values are stored in the `.grad` attribute of each parameter.

### The Accumulation Trap
PyTorch is designed to **add** (accumulate) gradients by default. If you don't clear them, the new gradients just get piled on top of the old ones from the previous round. This is why `optimizer.zero_grad()` or `model.zero_grad()` is mandatory in every loop.

---

## 🛠️ Proof of Work: Manual Gradient Tracker

I built a script that manually triggers the forward and backward passes to inspect exactly how gradients appear and how they accumulate if not reset.

| Script | What's inside |
| :--- | :--- |
| [**Manual_Gradient_Tracker.py**](./Manual_Gradient_Tracker.py) | Manual MSE calculation and gradient inspection logic. |

---

## 💡 Some Lessons Learned

### 1. The Scalar Requirement
You can't call `.backward()` on a vector or a matrix. The "Loss" must be reduced to a single **scalar** (one number). This is why we use Mean Squared Error or Average Cross Entropy—it squashes all the individual errors of a batch into one single value that the Autograd engine can handle.

### 2. The `.item()` Method
When you print a loss, it's a tensor attached to a huge graph. Using `loss.item()` pulls out the raw Python float. It’s essential for logging because it prevents your memory from filling up with old computational graphs you no longer need.

### 3. Gradient Flow isn't a Weight Update
A huge realization: `loss.backward()` **does not** change the weights. It only fills the "gradient" bucket. The weights stay exactly the same until an optimizer actually comes in and uses those gradients to make a move (which is the goal for Day 06).

---