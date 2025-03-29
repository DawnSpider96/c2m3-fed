## Install dependencies:

```bash
pip install -e .
```

### One-shot Merge

To run a one shot merge experiment, you can use the `MergeExperimentRunner` class from `c2m3.experiments.merge_experiment`:

```python
from c2m3.experiments.merge_experiment import MergeExperimentConfig, MergeExperimentRunner

# Configure the experiment
config = MergeExperimentConfig(
    experiment_name="my_merge_experiment",
    dataset_name="femnist",  # Options: 'femnist'
    model_name="cnn",  # Options: 'cnn'
    num_models=5,
    batch_size=64,
    learning_rate=0.01,
    data_distribution="iid",  # Options: 'iid', 'dirichlet', 'pathological', 'natural'
    initialization_type="identical",  # Options: 'identical' or 'random'
    merging_methods=["c2m3", "fedavg", "simple_avg", "median"]
)

runner = MergeExperimentRunner(config)
runner.setup()
runner.run()
runner.visualize_results()
```



### Parameterized Experiments (One-shot Merge)

There are also functions in the MergeExperiment.py file itself that allow for multiple experiments runs with one parameter being changed.

```python
from c2m3.experiments.merge_experiment import run_parameter_sweep, run_multiple_seeds_experiment

base_config = MergeExperimentConfig(
    experiment_name="merge_experiment_seeds",
    dataset_name="femnist",
    model_name="cnn",
    # ...
)
results = run_multiple_seeds_experiment(base_config, seeds=[42, 123, 456, 789, 101])

# Or sweep through different parameters
parameter_grid = {
    "learning_rate": [0.001, 0.01, 0.1],
    "non_iid_alpha": [0.1, 0.5, 1.0]
}
results = run_parameter_sweep(base_config, parameter_grid)
```

## Running Federated Experiments (Faulty setup)

Federated experiments simulate a federated learning scenario with multiple clients training on their local data and a server aggregating the models.

### Basic Usage

To run a federated learning experiment:

```python
from c2m3.experiments.federated_experiment_path import FederatedExperimentConfig, FederatedExperimentRunner

# Configure the experiment
config = FederatedExperimentConfig(
    experiment_name="my_federated_experiment",
    dataset_name="femnist",  # Options: 'femnist'
    model_name="cnn",  # Options: 'cnn'
    strategy_name="c2m3",  # Options: 'c2m3', 'fedavg'
    num_rounds=10,
    local_epochs=1,
    batch_size=64,
    learning_rate=0.01,
    data_distribution="iid",  # Options: 'iid', 'dirichlet', 'pathological', 'natural'
    num_total_clients=20,
    num_clients_per_round=5
)

# Run the experiment
runner = FederatedExperimentRunner(config)
runner.setup()
runner.run()
runner.visualize_results()
```

### Parameterized Experiments

For federated experiments with multiple parameter configurations:

```python
from c2m3.experiments.federated_experiment_path import run_parameter_sweep, run_multi_seed_learning_rates_experiment

# Run with multiple learning rates and seeds
base_config = FederatedExperimentConfig(
    experiment_name="federated_experiment_lr_seeds",
    dataset_name="femnist",
    model_name="cnn",
    # ...
)
results = run_multi_seed_learning_rates_experiment(
    base_config, 
    learning_rates=[0.001, 0.01, 0.1],
    seeds=[42, 123, 456, 789, 101]
)
```

## Notebooks

The repository contains several Jupyter notebooks in the `c2m3/notebooks` directory that provide proofs of concept and examples:

- `basic_flower.ipynb`: POC of implementing FrankWolfeSyncMerge (c2m3) in Flower
- `permute_test.ipynb`: POC of FrankWolfeSyncMatch (c2m3 but without model repair and merging)
- `fed_iid_lr0.01.ipynb`, `fed_iid_lr0.1.ipynb`, etc.: Federated learning experiments with different learning rates and data distributions.
