<h1 align="center"><img src="assets/MobyDick.png" width="30" height="30" alt="Enclose Moby Logo" /> Enclose Moby AI: Hybrid Optimization Solver & Procedural Generator</h1>
<p align="center">
  <img src="https://img.shields.io/badge/Python-3.8%2B-blue" alt="Python">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
  <img src="https://img.shields.io/badge/Status-Phase%203%20Completed-success" alt="Status">
</p>
<p align="center">
  <a href="https://kutluyigitturk.github.io/enclose.moby">
    <img src="https://img.shields.io/badge/▶_PLAY_NOW-1a1a2e?style=for-the-badge&logo=github&logoColor=white" alt="Play Now">
  </a>
</p>
<p align="center">
  <strong>Enclose Moby AI</strong> is an advanced algorithmic suite designed to solve and generate levels for the "Enclose Moby" puzzle game.
</p>
<p align="center">
  Unlike standard solvers that get stuck in local optima, this project utilizes a <strong>Hybrid Island-Memetic Architecture</strong> combining parallel evolution with local refinement to achieve <strong>100% optimal solutions</strong> on even the most complex maps.
</p>

---

## 🎮 About the Game

"Enclose Moby" is a puzzle game where players must trap a whale (Moby) by strategically placing walls (buoys) on a grid-based ocean map. The goal is to **maximize the enclosed area** while using a limited number of walls.

---

## 🚀 Key Features

### 1. Island Model Genetic Solver (The Brain)
- **4 Isolated Populations:** Instead of one mixed population, the AI maintains 4 separate "islands" that evolve independently, each focusing on different map regions.
- **Quadrant Specialization:** Each island is biased toward exploring solutions in a specific quadrant (NW, NE, SW, SE), preventing premature convergence to local optima.
- **No Migration Policy:** Islands do NOT share genetic material during evolution, ensuring diversity is preserved until the final comparison.

### 2. Memetic Optimizer (The Refiner)
- **Redundant Wall Detection:** Identifies walls that can be removed without allowing Moby to escape.
- **Drift Mutation (Shift):** Instead of random swaps, the optimizer tries shifting each wall by 1 tile in 4 directions, capturing micro-improvements that pure GA misses.
- **Post-Processing Refinement:** After GA converges, the optimizer polishes the solution to squeeze out every last point.

### 3. Procedural Map Generator (The Architect)
- **Chokepoint-Based Design:** Instead of random noise, the generator constructs maps based on strategic "chokepoints" and "rooms," mimicking human level design.
- **Solvability Validator:** Includes a built-in feedback loop. The generator creates a map, runs the solver in the background, and **only outputs the map if it is mathematically guaranteed to be solvable.**
- **Game-Ready Output:** Generates raw array data ready to be copy-pasted into Unity, Godot, or PyGame.

---

## 🧬 The Breakthrough: Island Model + Memetic Optimization

### The Problem: Local Optima Trap

Standard genetic algorithms kept converging to an **87-point solution** (local optimum) while the true optimal was **109 points**. The 87-point solution dominated early populations because it was "easy" to find, preventing exploration of better but harder-to-reach configurations.

```
Standard GA:                    Island Model:
                                
    Population                  Island 1 (NW)   Island 2 (NE)
        ↓                            ↓               ↓
   Converges to 87              Explores NW     Explores NE
        ↓                            ↓               ↓
   STUCK! 🔴                    Island 3 (SW)   Island 4 (SE)
                                     ↓               ↓
                                Explores SW     Explores SE
                                     ↓               ↓
                                BEST OF ALL → 109 ✅
```

### The Solution: Divide and Conquer

| Component | Strategy | Result |
|-----------|----------|--------|
| **Island Model** | 4 isolated populations, each biased to a map quadrant | Prevents 87-solution from dominating |
| **No Migration** | Islands never share genetic material | Preserves diversity |
| **Shift Mutation** | 90% shift (1 tile), 10% swap | Fine-grained exploration |
| **Memetic Optimizer** | Post-GA refinement with redundant wall relocation | +3 to +15 points improvement |

---

## 📁 Project Structure

```
enclose.mobyAi/
│
├── main.py              # CLI entry point
├── constants.py         # Game constants & test maps
├── models.py            # Data structures (MapState, EscapeAnalysis)
├── analysis.py          # Map analysis functions (flood fill, escape detection)
├── genetic_solver.py    # Island Model Genetic Algorithm
├── optimizer.py         # Memetic Optimizer (Drift + Redundancy Detection)
├── blind_solver.py      # Blind solving with saturation detection
├── generator.py         # Procedural map generator
├── visualization.py     # ASCII visualization & comparison tools
├── __init__.py          # Package exports
│
├── research_history/    # R&D Lab - The Engineering Journey
│   └── solver/
│       ├── v1_baseline_limitations.py   # Initial approach (stuck at 65)
│       └── v2_soft_penalty_solution.py  # Soft penalty breakthrough (68/68)
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

### 1. Generate & Solve a Random Map

Generates a unique, validated map and immediately attempts to solve it.

```bash
python main.py --generate
```

### 2. Solve the Classic "Pathfinder" Map

Runs the solver on the static benchmark map defined in the project.

```bash
python main.py
```

**Example Output:**
```
============================================================
BLIND SOLVER: ISLAND MODEL + MEMETIC OPTIMIZER
Strategy: 4 isolated populations + post-processing refinement
============================================================
Attempt 1: 95 | Best: 95 | 🚀 NEW RECORD
Attempt 2: 87 | Best: 95 | Stable (1/15)
Attempt 3: 106 | Best: 106 | 🚀 NEW RECORD
Attempt 4: 109 | Best: 109 | 🚀 NEW RECORD

🎉 TARGET REACHED! Found 109 points!
```

### 3. Additional Options

```bash
python main.py --patience 20      # Wait longer for improvements
python main.py --max-attempts 100 # Try more times
python main.py --target 109       # Stop when target is reached
python main.py --quiet            # Minimal output
```

### 4. Customize Map Generation

```bash
python main.py --generate --rows 20 --cols 25 --walls 15 --density 0.50
```

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--rows` | 22 | Map height |
| `--cols` | 19 | Map width |
| `--walls` | 12 | Number of buoys (wall budget) |
| `--density` | 0.55 | Land density (0.0 - 1.0) |

---

## 🧠 How It Works

### Phase 1: Soft Penalty Foundation

The original breakthrough that allowed the algorithm to escape zero-fitness dead zones.

```
If Moby is trapped:
    Fitness = 1,000,000 + Area
    
If Moby escapes:
    Fitness = -1,000,000 - (Escape_Points × 5000) - Area
```

### Phase 2: Procedural Generation

The generator creates strategic maps with guaranteed solvability:

1. **Chokepoint Placement** → Identifies narrow passages
2. **Structure Building** → Creates landmasses around chokepoints
3. **Validation Loop** → Only outputs maps the solver can solve

### Phase 3: Global Optimization (NEW)

The hybrid architecture that achieves 100% optimal solutions:

#### Island Model (Genetic Algorithm)
```python
# 4 Isolated Populations
islands = [
    create_population(bias="top_left"),      # Island 1: NW quadrant
    create_population(bias="top_right"),     # Island 2: NE quadrant
    create_population(bias="bottom_left"),   # Island 3: SW quadrant
    create_population(bias="bottom_right"),  # Island 4: SE quadrant
]

# Independent Evolution (NO MIGRATION)
for generation in range(150):
    for island in islands:
        evolve_independently(island)  # No mixing between islands!

# Final: Pick the global best from all islands
best = max(all_islands, key=fitness)
```

#### Memetic Optimizer (Local Refinement)
```python
def optimize(solution):
    # Step 1: Find redundant walls
    for wall in solution:
        if still_trapped_without(wall):
            relocate_to_better_position(wall)
    
    # Step 2: Shift optimization (drift)
    for wall in solution:
        for direction in [UP, DOWN, LEFT, RIGHT]:
            if shift_improves_area(wall, direction):
                apply_shift(wall, direction)
    
    return improved_solution
```

---

## 📊 Benchmark Results

### Pathfinder Map (22×19, 12 walls)

| Metric | Phase 2 (Old) | Phase 3 (New) |
|--------|---------------|---------------|
| Known Optimal | 109 | 109 |
| Our Best | 87 🔴 | **109** ✅ |
| Success Rate | ~60% | **100%** |
| Avg. Attempts | 15-20 | 3-5 |

### Performance Comparison

| Algorithm | Pathfinder Score | Time | Notes |
|-----------|------------------|------|-------|
| Random Search | ~45 | - | Baseline |
| Standard GA | 87 | ~10s | Stuck in local optimum |
| GA + Soft Penalty | 87-95 | ~10s | Sometimes escapes |
| **Island Model + Optimizer** | **109** | ~15s | **Always optimal** ✅ |

### Generated Maps

| Map Size | Walls | Avg. Solve Time | Success Rate |
|----------|-------|-----------------|--------------|
| 15×19 | 10 | ~5 sec | 100% |
| 20×25 | 14 | ~12 sec | 100% |
| 25×30 | 18 | ~25 sec | 100% |

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
- **Result:** Reaches 68/68 on simple maps ✅
- **Limitation:** Stuck at 87/109 on complex maps

### v3: Island Model + Memetic Optimizer (NEW)
- 4 isolated populations (no migration)
- Quadrant-biased initialization
- Shift mutation (90% drift, 10% swap)
- Post-processing redundancy detection
- **Result:** Consistently reaches 109/109 ✅

---

## 🗺️ Roadmap

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Genetic Solver (Soft Penalty) | ✅ Completed |
| 2 | Procedural Generator | ✅ Completed |
| 3 | Global Optimization (Island + Memetic) | ✅ Completed |
| 4 | Difficulty Metrics | 📋 Planned |
| 5 | MAP-Elites Integration | 📋 Planned |

### Upcoming Features
- **Deceptiveness Score:** Measure how "tricky" a map is
- **Solution Density:** Count alternative optimal solutions
- **MAP-Elites:** Generate diverse difficulty profiles
- **Web Interface:** Interactive solver visualization

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
- Island Model concepts from parallel evolutionary computation literature
- Special thanks to the procedural generation community

---

<p align="center">
  Made with ❤️ and lots of ☕
</p>