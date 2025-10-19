import os
import typing as tp

import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import PreTrainedTokenizer
from datasets import Dataset, DatasetDict


class CSVDataset:

    def __init__(
        self,
        csv_folder: str,
        input_column: str,
        target_column: str,
        train_ratio: float = 0.87,
        test_ratio: float = 0.06,
        random_seed: int = 42,
    ) -> None:
        self.csv_folder = csv_folder
        self.input_column = input_column
        self.target_column = target_column
        self.train_ratio = train_ratio
        self.test_ratio = test_ratio
        self.random_seed = random_seed

        self.full_df = None
        self.train_df = None
        self.val_df = None
        self.test_df = None

        self._load_csvs()
        self._split_dataset()

    def to_hf_dataset(
        self,
        tokenizer: tp.Optional[PreTrainedTokenizer] = None,
        max_input_length: int = 512,
        max_target_length: int = 512,
        tokenize: bool = True,
    ) -> DatasetDict:
        """
        Convert the train/val/test DataFrames to Hugging Face DatasetDict.
        """

        if self.train_df is None or self.val_df is None or self.test_df is None:
            raise ValueError("Dataset is not split yet. Call _split_dataset() first.")

        # Convert pandas DataFrames to HF Datasets
        hf_dataset = DatasetDict(
            {
                "train": Dataset.from_pandas(self.train_df),
                "validation": Dataset.from_pandas(self.val_df),
                "test": Dataset.from_pandas(self.test_df),
            }
        )

        if tokenizer is not None and tokenize:

            def tokenize_function(batch):
                model_inputs = tokenizer(
                    batch["source_text"], max_length=max_input_length, truncation=True
                )
                with tokenizer.as_target_tokenizer():
                    labels = tokenizer(
                        batch["target_text"],
                        max_length=max_target_length,
                        truncation=True,
                    )
                model_inputs["labels"] = labels["input_ids"]
                return model_inputs

            hf_dataset = hf_dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=["id", "source_text", "target_text"],
            )

        return hf_dataset

    def _load_csvs(self):
        """
        Load all CSV files from the folder, ensure each has 'source_text' and 'target_text',
        concatenate them into a single dataset, and add a unique 'id' column.
        """
        all_dfs = []
        for filename in os.listdir(self.csv_folder):
            if filename.endswith(".csv"):
                path = os.path.join(self.csv_folder, filename)
                df = pd.read_csv(path)

                # Ensure required columns exist
                for col in ["source_text", "target_text"]:
                    if col not in df.columns:
                        raise ValueError(
                            f"CSV file {filename} missing required column '{col}'"
                        )

                all_dfs.append(df[["source_text", "target_text"]])

        if not all_dfs:
            raise ValueError(f"No CSV files found in {self.csv_folder}")

        self.full_df = pd.concat(all_dfs, ignore_index=True)
        self.full_df.insert(0, "id", range(len(self.full_df)))

        print(
            f"[INFO] Loaded {len(self.full_df)} examples from {len(all_dfs)} CSV files."
        )

    def _split_dataset(self):
        """
        Split full_df into train, validation, and test sets deterministically using random_seed.
        """
        remaining_ratio: float = 1.0 - self.train_ratio
        val_ratio: float = remaining_ratio - self.test_ratio

        train_val_df, self.test_df = train_test_split(
            self.full_df,
            test_size=self.test_ratio,
            random_state=self.random_seed,
            shuffle=True,
        )

        self.train_df, self.val_df = train_test_split(
            train_val_df,
            test_size=val_ratio / (self.train_ratio + val_ratio),
            random_state=self.random_seed,
            shuffle=True,
        )

        print(
            f"[INFO] Train examples: {len(self.train_df)}, "
            f"Validation examples: {len(self.val_df)}, "
            f"Test examples: {len(self.test_df)}"
        )

    def save_to_csv(self, output_folder: str = "./datasets/splits"):
        """
        Save train/validation/test sets as CSV files.
        """
        os.makedirs(output_folder, exist_ok=True)

        self.train_df.to_csv(os.path.join(output_folder, "train.csv"), index=False)
        self.val_df.to_csv(os.path.join(output_folder, "val.csv"), index=False)
        self.test_df.to_csv(os.path.join(output_folder, "test.csv"), index=False)

        print(f"[INFO] Saved train/validation/test CSV files to {output_folder}")

    def get_splits(self):
        """
        Return train, validation, and test DataFrames.
        """
        return self.train_df, self.val_df, self.test_df

    def sample(self, n: int = 5):
        """
        Print sample rows from the dataset.
        """
        print("[INFO] Sample data:")
        print(self.full_df.sample(n))


if __name__ == "__main__":
    from transformers import T5Tokenizer

    tokenizer = T5Tokenizer.from_pretrained("t5-small")

    dataset = CSVDataset(
        csv_folder="../data/training",
        input_column="source_text",
        target_column="target_text",
    )

    dataset.save_to_csv()

    hf_dataset = dataset.to_hf_dataset(tokenizer=tokenizer)

    print(hf_dataset)
