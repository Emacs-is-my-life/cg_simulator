# Overview
## Introduction
* Discrete-Event System based LLM inference simulator, using **Compute Graph**
* Testbed for various data placement strategies.

## How to use
```bash
$ pip install -e   # Package Installation
$ python main.py --not-defined-yet goo goo ga ga
```

## Assumptions
* Model is compiled to a **Compute Graph**  
* **Tensor** is an atomic unit of data  
  - **Tensor** is laid out in virtual memory as continuous pages  
  - A **Tensor** belongs to one of: **Input**, **Weight**, **Intermediate** or **KV cache** category  
* Compute can only process a **Node** when **Tensor**s required to process it are in memory  
* Data in memory/disk are managed in page granularity
* Disk has an infinite size

# Code Structure
## Simulator Modules
### `src/cg_sim/`
- Read input(model, hw config), initialize each components in simulator 
- Data structures representing the workload(**Compute Graph**) and memory state
- Core simulation logic managing simulation time and the event queue  
- Bookkeeping of each **Tensor** for correct simulation
  - Isn't scheduler trying to use more memory than available?
  - Is required **Tensor** present in memory, before the compute starts execution of a **Node**?
  - Is space in memory secured for loading a **Tensor** from disk?
- Hardware modeling for computation and IO latency estimation
- Log event traces for research

### `src/cg_sched/`
- Manages data placement and **Node**(in **Compute Graph**) execution scheduling  
- Functionally equivalent to {OS + Inference Engine(llama.cpp, ...) + Cache/Prefetch logic}  
  - LRU page management of Linux
  - Memory locking & prefetching like FlexInfer
- Modular architecture: Easy to write your own scheduler logic for the simulator  

### `main.py`
- Simulator entry point

## Trivia
### `input/`
- LLM Model profiling records for simulation  
- Hardware configuration  
  
### `output/`
- For simulation output  

### `scripts/`
- Scripts for running experiments
- Visualization code
