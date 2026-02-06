## Robustness Verification of Binary Neural Networks: An Ising and Quantum-Inspired Framework

This repository implements an end-to-end pipeline for **robustness verification of binary neural networks (BNNs)** by mapping the verification problem to a **Quadratic Unconstrained Binary Optimization (QUBO)** instance. Starting from a trained ten-class BNN and a correctly classified input, converts it into a dense QUBO using penalty methods, and searches for adversarial perturbations within a prescribed budget.

The framework is designed to work with both **conventional algorithms** and **unconventional Ising-style hardware**.

The goal of this codebase is **demonstrate that BNN robustness verfication problem can be expressed as QUBO instances which cab solved on both classical and emerging Ising/annealing platforms**, providing a bridge between AI trustworthiness and unconventional computing.
