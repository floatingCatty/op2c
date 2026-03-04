# Feature Development Workflow

This document outlines the standardized workflow for adding new features to `op2c`. It is designed to ensure robust design, high performance, and seamless integration with the existing codebase.

## Phase 1: Requirement & Specification
**Input**: User/Developer provides a high-level goal or requirement.

1.  **Define the Scope**: clearly state *what* the feature is and *why* it is needed.
2.  **Identify Inputs/Outputs**: Only high-level data flow at this stage.
3.  **Constraints**: Performance requirements (e.g., O(N) scaling), external dependencies, hardware targets (CPU/GPU).

## Phase 2: Reference Research & Acquisition
**Trigger**: If no reference implementation is provided or known.

1.  **Literature Search**: Search for standard algoithms or libraries that implement similar features.
    *   *Example*: "How do SIESTA/OpenMX handle grid integration?"
2.  **Code Discovery**: Identify open-source repositories (GitHub/GitLab) that implement these methods.
3.  **Acquisition**: Download the codebase for analysis.
    *   *Action*: `git clone <repo_url>` to a temporary or vendor directory.
    *   *Goal*: Obtain source code to study "Gold Standard" implementations.

## Phase 3: Deep Analysis (`/readcode`)
**Input**: The `op2c` codebase + Acquired Reference Code.

1.  **Internal Codebase**: Use the `readcode` workflow to analyze relevant existing `op2c` modules.
    *   Find where the new feature should live.
    *   Identify existing data structures to reuse (avoid duplication).
2.  **Reference Analysis**: Analyze the acquired reference code (from Phase 2).
    *   Understand their memory model (e.g., Row-major vs Col-major, MPI distribution).
    *   Identify key algorithms and data structures.
3.  **Gap Analysis**: Precisely identify what is missing in `op2c` to support the new feature compared to the reference.

## Phase 4: Comparative Research & Selection
**Goal**: Select the *best* method, not just the first one found.

1.  **Methodology Comparison**: Compare algorithms from different references (e.g., Cubic Splines vs Linear Interpolation).
2.  **Architecture Comparison**: How do they structure their data? (e.g., Distributed Block-Cyclic vs 1D Element distribution).
3.  **Performance Check**: What optimization techniques are standard? (e.g., Symmetry utilization, SIMD).

**Outcome**: A decision on the standard method to implement, backed by evidence.

## Phase 5: Design & Architecture
Draft the technical design document.

1.  **Component Design**: Break the feature into smaller, independent classes.
    *   *Principle*: "Intrinsic Properties". Logic should reside where the data is (e.g., Basis sets evaluate themselves).
2.  **Interface Definition**: Define abstract C++ interfaces (`virtual` classes) to decouple components.
3.  **Data Flow**: Diagram how data moves between components.

## Phase 6: Implementation Planning
Create a step-by-step plan (`implementation_plan.md`).

1.  **Phasing**: Break work into logical phases.
    *   *Phase 1: Foundation* (Core data structures, Math helpers).
    *   *Phase 2: Core Logic* (The main algorithm).
    *   *Phase 3: Integration* (Connecting to existing code, Python bindings).
2.  **Task List**: Create granular tasks in `task.md`.

## Phase 7: Iterative Implementation
Build and verify in loops.

1.  **Prototype**: Implement the core logic first (even if inefficient) to verify correctness.
2.  **Test**: Write unit tests (GoogleTest for C++, `pytest` for Python) immediately.
3.  **Optimize**: Apply performance optimizations (OpenMP/MPI, vectorization) once correctness is proven.
4.  **Bind**: Expose to Python via `pybind11`.
