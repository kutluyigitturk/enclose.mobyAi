# 🧪 Solver Research History (R&D Log)

This folder documents the evolutionary process of the optimization algorithms developed to solve the "Enclose Moby" problem.

The code contained here serves as a technical case study demonstrating how we transitioned from a standard approach trapped in a **Local Optimum** to a customized solution that reached the **Global Optimum (68 Points)**.

---

## 📂 Files and Experiments

### 1. `v1_genetic_approach_limitations.py` (Failed Attempt)
* **Status:** Stuck in Local Optimum.
* **Max Score:** 64-65 Points.
* **Method Used:**
  * **Hard Penalty:** The solution score was dropped to negative the moment Moby could escape.
  * **Edge Snap:** Strategy to randomly snap walls to the grid edges.
* **Analysis & Lessons:**
  * This approach prevented the algorithm from "taking risks".
  * To go from 65 to 68 points, the existing wall structure needed to be temporarily broken (resulting in a temporary score loss).
  * Due to the hard penalty mechanism, the algorithm could not cross this **"Valley of Death"** (where the score drops) and stopped evolving.

### 2. `v2_soft_penalty_solution.py` (Successful Solution)
* **Status:** Global Optimum Found (68/68).
* **Max Score:** 68 Points.
* **Method Used:**
  * **Soft Penalty:** Even if Moby escapes, partial points are awarded to the algorithm if the escape routes (frontiers) are reduced.
  * **Stochastic Path Blocking:** Instead of random mutation, Moby's current escape route is detected (via BFS) and a wall is placed directly on that path.
  * **Geometry Mutation:** Clustering of walls (forming blocks) is encouraged.
* **Result:**
  * Thanks to the "Soft Penalty", the algorithm was not afraid to generate temporarily bad solutions in order to build a better structure.
  * The stochastic path blocking feature produced "guided" mutations rather than blind searches.
  * Consequently, the ideal solution of 68 points was found.

---

## 📊 Comparison Table

| Feature | v1 (Baseline) | v2 (Advanced) |
| :--- | :--- | :--- |
| **Penalty Mechanism** | **Hard:** Escape = 0 Points | **Soft:** Escape = Low Score (Evolution continues) |
| **Mutation Type** | Random / Edge Snapping | Smart Path Blocking |
| **Behavior** | Conservative (Takes no risks) | Explorer (Takes risks, crosses the valley) |
| **Result** | 65 Points (Stable) | **68 Points (Global Optimum)** |

---

## 🚀 How to Run?

These codes are standalone and can run independently. You can execute the relevant file in the terminal to test:

```bash
# To see the failed scenario:
python v1_genetic_approach_limitations.py

# To see the successful scenario:
python v2_soft_penalty_solution.py