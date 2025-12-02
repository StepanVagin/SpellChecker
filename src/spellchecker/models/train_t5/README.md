# How to Train the T5 Model

### 1. Installation
Before configuring or running the pipeline, ensure you have the specific dependencies installed for the T5 training module. Run the following command from the project root:

```bash
pip install -r src/spellchecker/models/train_t5/requirements.txt
```

### 2. Configuration
Open the configuration file located at:
`src/spellchecker/models/conf/t5_train_cfg.yaml`

Modify the parameters as needed. Below is the explanation for each setting:

```yaml
paths:
  csv_folder: ../data/training      # Path to the directory containing input CSV files for training
  test_csv: ./datasets/splits/test.csv # Path to the specific CSV file used for final testing
  model_name: /root/t5-small        # Path to the pre-trained checkpoint or Hugging Face model ID (e.g., 't5-small')
  output_dir: ./results             # Directory where model checkpoints and logs will be saved
  predictions_csv: ./results/predictions.csv # File path where inference results will be saved

train:
  num_train_epochs: 4               # Total number of training cycles over the dataset
  per_device_train_batch_size: 32   # Batch size per GPU/CPU for training
  per_device_eval_batch_size: 32    # Batch size per GPU/CPU for evaluation
  fp16: true                        # Enable mixed precision training (saves memory and speeds up training on modern GPUs)
  learning_rate: 4e-4               # Initial learning rate for the optimizer
  weight_decay: 1e-3                # Regularization parameter to prevent overfitting
  max_length: 512                   # Maximum token length for the input sequences
  logging_steps: 50                 # Number of steps between logging events
  save_steps: 500                   # Number of steps between saving model checkpoints
  eval_steps: 50                    # Number of steps between evaluation runs
  save_total_limit: 2               # Maximum number of checkpoints to keep (older ones are deleted)
  seed: 42                          # Random seed for reproducibility
  evaluation_strategy: steps        # Determines when to evaluate (can be 'steps' or 'epoch')
  gradient_accumulation_steps: 64   # Number of steps to accumulate gradients before performing a backward/update pass
  max_grad_norm: 1                  # Maximum gradient norm for gradient clipping
  warmup_ratio: 0.1                 # Ratio of total training steps used for a linear warmup
  predict_with_generate: false      # Whether to use generate() for calculating generative metrics during evaluation
  remove_unused_columns: true       # Remove columns from the dataset that are not required by the model
  report_to: all                    # List of integrations to report results to (e.g., 'wandb', 'tensorboard', or 'all')
  metric_for_best_model: loss       # The metric used to compare models and determine the best one
  greater_is_better: false          # Boolean indicating if a higher metric value is better (false for loss)
  load_best_model_at_end: true      # Whether to load the best model (according to the metric) at the end of training

inference:
  batch_size: 32                    # Batch size used for the inference stage
  max_length: 512                   # Maximum sequence length regarding token generation
  device: cuda                      # Device to run inference on ('cuda' for GPU or 'cpu')
  text_column: source_text          # The name of the column in the CSV containing the input text

metrics:
  enabled: true                     # Enable or disable metric calculation after inference
```

### 3. Execution
Once the dependencies are installed and the configuration is set, navigate to the source directory and run the training pipeline module.

```bash
cd src
python -m spellchecker.models.train_t5.pipeline
```

### 4. Pipeline Overview
The automated pipeline performs the following actions sequentially:
1.  **Workspace Setup:** Automatically creates a unique folder for the current training instance.
2.  **Data Splitting:** Splits the source data into training, validation, and testing folds.
3.  **Training:** Trains the T5 model based on the parameters defined in the YAML file.
4.  **Inference:** Runs the trained model on the test dataset.
5.  **Evaluation:** Calculates performance metrics based on the inference results.
6.  **Logging:** Saves all logs, configurations, and artifacts into the corresponding output folder.