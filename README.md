# pso-camopatch
An extension of WIlliams and Li's CamoPatch (2023) methodology for evolutionary optimization of camouflaged adversarial image patches.

## Project Goal

This project explores an alternative method for creating camouflaged adversarial patches, as introduced by Williams & Li. [Link](https://github.com/phoenixwilliams/CamoPatch) The original paper used an Evolutionary Strategy with Simulated Annealing to generate these patches, which are made of semi-transparent circles designed to fool image recognition systems while being hard to see. While the computational art framework is innovative and effective, the method may be limited by the use of a single-trajactory search strategy, particularly when trying to simultaneously optimize patch content and placement.

This notebook implements and tests a custom PSO implementation with a similar computational art inspiration. The aim is to see if PSO can find effective, low-visibility patches more efficiently than the methods used in the original paper and other standard techniques.


## How to Use This Notebook

1.  **Environment:** This notebook was designed to be portable, and runs in most notebook environments. A GPU runtime is strongly recommended for faster model inference, but a CPU will work.

2.  **Libraries:** The code uses de facto standard Python libraries readily available in Google Colab. No extra installations are needed. Test images are pulled from an online sample repository by default for portability, but local images can easily be used so long as they are uploaded to session storage.

3.  **Running the Cells:** Execute the cells in order from top to bottom.
    * **Cell 1:** Imports required libraries.
    * **Cells 2-7:** Define necessary functions and classes for patch generation, evaluation, PSO, and plotting.
    * **Cell 8:** Defines the main `run()` function to execute the PSO algorithm.
    * **Cell 9:** Sets parameters and runs the main PSO experiment. This is the core experiment.
    * **Appendix A (Optional):** Runs a sensitivity analysis for selected PSO hyperparameters.
    * **Appendix B (Optional):** Runs a Random Search baseline for comparison.
    * **Appendix C (Optional):** Runs a Stochastic Hill Climbing baseline for comparison.
    * **Appendix D (Optional):** Generates the averaged convergence plot (Figure 2 in the report) using saved data from multiple runs.
    * **Appendix E (Optional):** Generates patch placement comparison plots (Figure 5 in the report).
    * **Appendix F (Optional):** Calculates statistics, performs tests, and generates the summary results table (Table 1 in the report).

4.  **Reproducibility:** The notebook is designed for reproducibility. Running Cell 9 executes the main PSO experiment. Appendices A-F run the other analyses and comparisons independently. Clear instructions are provided throughout.

5.  **Outputs:** Each experiment run will:
    * Print progress updates.
    * Show a summary of the best patch found.
    * Display a figure comparing the original image, the best patch, and the modified image.
    * Show a plot illustrating how the solution improved over time (convergence plot).