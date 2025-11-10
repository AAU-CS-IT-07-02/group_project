**List of possible alternative methods for data-driven modelling**

First of all, I'd recommend using the **PyCaret** python library, as it
**supports several data-driven modelling technique**s, making
comparisons easier. In a nutshell, it is a **wrapper around several
machine learning libraries and frameworks**, such as **scikit-learn**,
XGBoost, LightGBM, CatBoost, spaCy, Optuna, Hyperopt, Ray and a few more
(see <https://pycaret.gitbook.io/docs>).

Here is the list of possible alternative methods, drawing inspiration
from a paper on Energy Efficiency Smart Buildings Models[^1]:

-   Multi-Layer Perceptron (MLP) (on PyCaret we could use create_model('mlp') for example)

-   Support Vector Machines with Radial Basis Function Kernel (SVM) ('svm')

-   Gaussian Process with Radial Basis Function Kernel (Gauss) ('gaussian_process')

-   Bayesian Regularized Neural Networks (BRNN) (it doesn't exist)

-   Random forest (RF) ('rf')

From these methods, one must be chosen. Experiments must be conducted to
draw conclusions.

[^1]: Aurora González-Vidal, Victoria Moreno-Cano, Fernando
    Terroso-Sáenz, Antonio F. Skarmeta, *Towards Energy Efficiency Smart
    Buildings Models Based on Intelligent Data Analytics*, Procedia
    Computer Science, Volume 83, 2016, Pages 994-999, ISSN 1877-0509,
    <https://doi.org/10.1016/j.procs.2016.04.213>.
