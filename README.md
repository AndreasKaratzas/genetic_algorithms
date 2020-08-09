# Genetic Algorithm using Numpy in Python

### Movie Recommendation System implemented from scratch

![Genetic Algorithms](https://biology.mit.edu/wp-content/uploads/2017/12/Cell-karyotype-exhibiting-trisomy_Credit_NIH.jpeg)

In this project, the [MovieLens](https://grouplens.org/datasets/movielens/100k/) dataset was used to optimize movie recommendation using genetic algorithms. The code is available for Pycharm users. There are also Jupyter notebooks that contain the same code, that help those that use Jupyter or Jupyter Lab. The repo is organised as follows:
* `Deep-Learning-GA-Universe.ipynb`: Jupyter notebook that has the complete implementation of the algorithm
* `Deep-Learning-GA-holdout.ipynb`: Jupyter notebook uses the holdout validation techique for genetic algorithms
* `Deep-Learning-GA-proof.ipynb`: Jupyter notebook that serves the testing purposes of the developer
* `Deep-Learning-GA.ipynb`: Jupyter notebook that computes chromosomes only for the first user
* `Deep-Learning_GA_CPU.py`: Python script that uses the CPU for all operations
* `Deep-Learning_GA_GPU.py`: Python script that uses the GPU for all operations
* `u.data`: The dataset used in the project

The rest of the files are some of the results of the scripts and extra code that was used by the developer.  

There are various genetic algorithm techniques implemented for the purposes of the project:
* For choromosome selection:
    * Roulette wheel selection
    * Rank selection
    * Tournament selection
* For selected chromosomes crossover:
    * Single point crossover
    * Multiple point crossover
    * Uniform crossover
* For chromosome mutation:
    * Elitism with both:
        * Gauss replacement option
        * Reset replacement option
    * Random mutation with both:
        * Gauss replacement option
        * Reset replacement option
* For chromosome evaluation:
    * Pearson correlation coefficient
    * Custom method was also implemented

The final population is saved in `.txt` form. In function `initialize_hyperparameters`, anyone can initialize custom genetic algorithm options. 
