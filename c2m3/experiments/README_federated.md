# Federated Learning Experiment Framework

This framework allows you to run federated learning experiments with configurable parameters such as aggregation frequencies and local learning rates. It is built on top of the [Flower](https://flower.dev/) federated learning framework.

## Overview

The federated learning experiment framework consists of:

1. **FederatedExperimentConfig**: A dataclass containing all configuration parameters for experiments
2. **FlowerRayClient**: A Flower client implementation for federated learning
3. **FederatedExperimentRunner**: The main class for setting up and running experiments
4. **Parameter sweep functions**: Helper functions for running experiments with different parameter settings

## Running Experiments

### Basic Usage

To run a simple federated learning experiment:

```bash
python -m c2m3.experiments.run_federated_experiment \
    --experiment-name "fed_test_femnist" \
    --dataset femnist \
    --model cnn \
    --num-rounds 10 \
    --local-epochs 1 \
    --learning-rate 0.01 \
    --server-learning-rate 1.0 \
    --num-total-clients 20 \
    --num-clients-per-round 5 \
    --data-distribution natural
```

### Parameter Sweeps

You can also run parameter sweeps to test the effect of different parameter values:

#### Learning Rate Sweep

```bash
python -m c2m3.experiments.run_federated_experiment \
    --experiment-name "fed_lr_sweep" \
    --dataset femnist \
    --model cnn \
    --experiment-type learning-rate \
    --learning-rates 0.001,0.01,0.05,0.1
```

#### Server Learning Rate Sweep

```bash
python -m c2m3.experiments.run_federated_experiment \
    --experiment-name "fed_server_lr_sweep" \
    --dataset femnist \
    --model cnn \
    --experiment-type server-lr \
    --server-learning-rates 0.1,0.5,1.0,2.0
```

#### Aggregation Frequency Sweep

```bash
python -m c2m3.experiments.run_federated_experiment \
    --experiment-name "fed_agg_freq_sweep" \
    --dataset femnist \
    --model cnn \
    --experiment-type aggregation-frequency \
    --aggregation-frequencies 1,2,3,5
```

#### Local Epochs Sweep

```bash
python -m c2m3.experiments.run_federated_experiment \
    --experiment-name "fed_local_epochs_sweep" \
    --dataset femnist \
    --model cnn \
    --experiment-type local-epochs \
    --local-epoch-values 1,2,5,10
```

#### Multiple Seeds (Reproducibility)

```bash
python -m c2m3.experiments.run_federated_experiment \
    --experiment-name "fed_multi_seed" \
    --dataset femnist \
    --model cnn \
    --experiment-type multi-seed \
    --seeds 42,123,456,789,101
```

## Key Parameters

- **experiment_name**: Unique name for the experiment
- **dataset_name**: Dataset to use (currently 'femnist')
- **model_name**: Model architecture to use (currently 'cnn')
- **num_rounds**: Number of federation rounds
- **local_epochs**: Number of local training epochs per round
- **learning_rate**: Client learning rate
- **server_learning_rate**: Server learning rate for FedAvg
- **aggregation_frequency**: How often to aggregate models (every X rounds)
- **num_total_clients**: Total number of available clients
- **num_clients_per_round**: Number of clients sampled per round
- **data_distribution**: How data is distributed ('iid', 'dirichlet', 'pathological', 'natural')

## Output

Experiment results are saved in the specified output directory (default: './results') with:

1. **config.json**: The full experiment configuration
2. **results.json**: Detailed results for all rounds
3. **accuracy_plot.png**: Plot of model accuracy over rounds
4. **train_loss_plot.png**: Plot of training loss over rounds (if available)
5. **flower_history.pkl**: Pickled Flower history object 
6. **final_model.pt**: Final model weights (if save_models is enabled)

For parameter sweeps, additional files are generated to compare performance across different parameter values.

## Example Code

To run experiments programmatically:

```python
from c2m3.experiments.federated_experiment import (
    FederatedExperimentConfig, 
    FederatedExperimentRunner,
    run_learning_rate_experiment
)

# Create configuration
config = FederatedExperimentConfig(
    experiment_name="federated_experiment",
    dataset_name="femnist",
    model_name="cnn",
    num_rounds=10,
    local_epochs=1,
    learning_rate=0.01,
    server_learning_rate=1.0,
    data_distribution="natural"
)

# Run a single experiment
runner = FederatedExperimentRunner(config)
runner.setup()
results = runner.run()
runner.visualize_results()

# Or run a parameter sweep
run_learning_rate_experiment(config, [0.001, 0.01, 0.1])
``` 