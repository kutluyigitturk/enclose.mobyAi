# <img src="assets/MobyDick.png" width="40" height="40" alt="Enclose Moby Logo" /> Enclose Moby AI: Hybrid Optimization Solver & Procedural Generator

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Phase%202%20Completed-success)

**Enclose Moby AI** is an advanced algorithmic suite designed to solve and generate levels for the "Enclose Moby" puzzle game.

Unlike standard solvers that get stuck in local optima, this project utilizes a **Genetic Algorithm with Soft Penalty** logic to find global maximums (perfect traps). Furthermore, it features a **Procedural Level Generator** capable of designing strategic, guaranteed-solvable maps for game engines.

---

## 🎮 About the Game

"Enclose Moby" is a puzzle game where players must trap a whale (Moby) by strategically placing walls (buoys) on a grid-based ocean map. The goal is to **maximize the enclosed area** while using a limited number of walls.

<p align="center">
  <img src="assets/gameplay.png" alt="Gameplay Screenshot" width="400">
</p>

---

## 🚀 Key Features

### 1. The Genetic Solver (The Brain)
- **Soft Penalty Architecture:** Solves the "Valley of Death" problem where standard algorithms fail. Even if Moby escapes, the algorithm rewards "almost successful" trap configurations to evolve better solutions.
- **Stochastic Path Blocking:** Instead of random walls, the AI analyzes Moby's escape route and strategically blocks chokepoints.
- **Blind Solving:** Can solve unknown maps without knowing the target optimal score using statistical saturation.

### 2. Procedural Map Generator (The Architect)
- **Chokepoint-Based Design:** Instead of random noise, the generator constructs maps based on strategic "chokepoints" and "rooms," mimicking human level design.
- **Solvability Validator:** Includes a built-in feedback loop. The generator creates a map, runs the solver in the background, and **only outputs the map if it is mathematically guaranteed to be solvable.**
- **Game-Ready Output:** Generates raw array data ready to be copy-pasted into Unity, Godot, or PyGame.

---

## 📁 Project Structure

```
enclose.mobyAi/
│
├── main.py              # CLI entry point
├── constants.py         # Game constants & test maps
├── models.py            # Data structures (MapState, EscapeAnalysis)
├── analysis.py          # Map analysis functions (flood fill, escape detection)
├── genetic_solver.py    # Genetic Algorithm implementation
├── blind_solver.py      # Blind solving with saturation detection
├── generator.py         # Procedural map generator
├── visualization.py     # ASCII visualization & comparison tools
├── __init__.py          # Package exports
│
├── research_history/    # R&D Lab - The Engineering Journey
│   └── solver/
│       ├── v1_baseline_limitations.py   # Initial approach (stuck at 65)
│       └── v2_soft_penalty_solution.py  # Breakthrough solution (68/68)
│
└── README.md
```

---

## 🛠️ Installation

```bash
# Clone the repository
git clone https://github.com/kutluyigitturk/enclose.mobyAi.git

# Navigate to the directory
cd enclose.mobyAi

# Install dependencies (only numpy is required)
pip install numpy
```

---

## 💻 Usage

The `main.py` serves as the CLI entry point for both solving and generating maps.

### 1. Generate & Solve a Random Map (New!)

Generates a unique, validated map and immediately attempts to solve it.

```bash
python main.py --generate
```

**Example Output:**
```
🌍 GENERATING MAP (22x19)...

==================================================
📋 GAME READY DATA (Copy & Paste)
==================================================
GENERATED_MAP = [
    [0, 0, 1, 1, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0, 0],
    ...
]
# Moby (3) placed into the grid at position (9, 7).
==================================================

🧠 SOLVER ENGAGING...
Attempt 1: 45 | Best: 45 | 🚀 NEW RECORD
Attempt 2: 52 | Best: 52 | 🚀 NEW RECORD
...
✅ SATURATION REACHED: Could not beat 67 in the last 15 attempts.
```

### 2. Customize Map Generation

Create a larger or harder map by adjusting dimensions and wall budget.

```bash
python main.py --generate --rows 20 --cols 25 --walls 15 --density 0.50
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--rows` | 22 | Map height |
| `--cols` | 19 | Map width |
| `--walls` | 12 | Number of buoys (wall budget) |
| `--density` | 0.55 | Land density (0.0 - 1.0) |

### 3. Solve the Classic "Pathfinder" Map

Runs the solver on the static benchmark map defined in the project.

```bash
python main.py
```

### 4. Additional Options

```bash
python main.py --patience 20      # Wait longer for improvements
python main.py --max-attempts 100 # Try more times
python main.py --target 68        # Stop when target is reached
python main.py --quiet            # Minimal output
```

---

## 🧠 How It Works

### Phase 1: The Solver Logic

The solver uses an evolutionary approach. A population of 300+ "wall configurations" evolves over generations.

| Component | Description |
|-----------|-------------|
| **Fitness Function** | `Score = TrapBonus + (SoftPenalty if Escape)` |
| **Mutation** | 40% Path Block, 30% Geometry, 30% Random |
| **Selection** | Tournament Selection (size=5) |
| **Crossover** | Single-point crossover with deduplication |

**The Soft Penalty Breakthrough:**

Standard algorithms fail because they give `0` fitness to escaped states. Our approach:

```
If Moby is trapped:
    Score = 20,000 + (Area × 100)
    
If Moby escapes:
    Score = 5,000 + (Blocked Area × 10) - (Escape Points × 500)
```

This allows the algorithm to "climb out" of local optima by rewarding partial progress.

### Phase 2: The Generator Logic

The generator does not rely on pure randomness.

1. **Chokepoint Placement:** Identifies potential narrow passages (minimum 4 cells apart)
2. **Structure Building:** Creates landmasses around chokepoints
3. **Edge Blocks:** Adds strategic blocks on map edges
4. **Connectivity Check:** Ensures all water cells are connected
5. **Validation Loop:**

```
┌─────────────────────────────────────────────────────┐
│  Generator: "I made a map."                         │
│      ↓                                              │
│  Solver: "Let me check... I can solve it!"          │
│      ↓                                              │
│  Generator: "Great! Here's your game-ready data."   │
└─────────────────────────────────────────────────────┘
```

This loop ensures **100% playability** for the end user.

---

## 📊 Benchmark Results

### Pathfinder Map (15×19, 10 walls)

| Metric | Value |
|--------|-------|
| Known Optimal | 68 |
| Our Best | **68** ✅ |
| Success Rate | ~95% (reaches 68 within 20 attempts) |
| Avg. Attempts | 8-12 |

### Generated Maps

| Map Size | Walls | Avg. Solve Time | Success Rate |
|----------|-------|-----------------|--------------|
| 15×19 | 10 | ~5 sec | 100% |
| 20×25 | 14 | ~12 sec | 100% |
| 25×30 | 18 | ~25 sec | 98% |

---

## 🔬 Research History

The `research_history/` folder documents the engineering journey:

### v1: Baseline Approach
- Standard genetic algorithm
- **Problem:** Stuck at 65/68 (local optimum)
- **Cause:** Zero fitness for escape = no evolution gradient

### v2: Soft Penalty Solution
- Added soft penalty for partial success
- Stochastic path analysis
- **Result:** Consistently reaches 68/68 ✅

---

## 🗺️ Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Genetic Solver | ✅ Completed |
| 2 | Procedural Generator | ✅ Completed |
| 3 | Difficulty Metrics | 🔄 In Progress |
| 4 | MAP-Elites Integration | ⏳ Planned |

### Upcoming Features
- **Deceptiveness Score:** Measure how "tricky" a map is
- **Solution Density:** Count alternative optimal solutions
- **MAP-Elites:** Generate diverse difficulty profiles

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 👤 Author

**Kutlu Yiğittürk**

- GitHub: [@kutluyigitturk](https://github.com/kutluyigitturk)

---

## 🙏 Acknowledgments

- Inspired by classic puzzle games like Sokoban and Flow Free
- Genetic Algorithm concepts from "Introduction to Evolutionary Computing" by A.E. Eiben
- Special thanks to the procedural generation community

---

<p align="center">
  Made with ❤️ and lots of ☕
</p>
