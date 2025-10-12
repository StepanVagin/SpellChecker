import os
import typing as tp
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from tqdm import tqdm

from spellchecker.data.processors.utils import query_llm


@dataclass
class LLMProcessor:
    """Generic LLM processor for dataframe columns."""

    prompt_template: str
    model_name: str
    output_column: str
    batch_size: int = 64
    checkpoint_dir: Path = Path("checkpoints")
    temperature: float = 0.7
    max_tokens: int = 500

    def __post_init__(self):
        self.checkpoint_dir.mkdir(exist_ok=True)

    def process_row(self, row: pd.Series, **kwargs) -> str:
        """Process single row with LLM. Override for custom logic."""
        prompt = self.prompt_template.format(**row.to_dict(), **kwargs)
        response = "".join(
            query_llm(
                prompt,
                model=self.model_name,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )
        )
        return response.strip()

    def process_dataframe(
        self,
        data: tp.Union[pd.DataFrame, str, Path],
        input_columns: tp.Optional[tp.List[str]] = None,
        **process_kwargs,
    ) -> pd.DataFrame:
        """Process dataframe with checkpointing."""
        # Load data
        if isinstance(data, (str, Path)):
            df = pd.read_csv(data)
            input_name = Path(data).stem
        else:
            df = data.copy()
            input_name = "dataset"

        # Validate columns
        if input_columns:
            assert set(input_columns).issubset(
                df.columns
            ), f"Missing required columns: {set(input_columns) - set(df.columns)}"

        # Setup checkpoint
        checkpoint_path = self.checkpoint_dir / f"{input_name}_{self.output_column}.csv"

        # Resume from checkpoint
        start_idx = 0
        if checkpoint_path.exists():
            existing_df = pd.read_csv(checkpoint_path)
            if self.output_column in existing_df.columns and len(existing_df) == len(
                df
            ):
                processed_mask = existing_df[self.output_column].notna()
                df[self.output_column] = existing_df[self.output_column]
                start_idx = processed_mask.sum()
                print(f"Resuming from checkpoint: {start_idx}/{len(df)} rows processed")

        if self.output_column not in df.columns:
            df[self.output_column] = pd.NA

        # Process rows in batches
        for i in tqdm(
            range(start_idx, len(df), self.batch_size),
            desc=f"Processing {self.output_column}",
        ):
            batch_end = min(i + self.batch_size, len(df))

            for idx in range(i, batch_end):
                try:
                    df.loc[idx, self.output_column] = self.process_row(
                        df.loc[idx], **process_kwargs
                    )
                except Exception as e:
                    print(f"Error processing row {idx}: {e}")
                    df.loc[idx, self.output_column] = None

            # Save checkpoint
            df.to_csv(checkpoint_path, index=False)

        return df

    def process_multiple(
        self, datasets: tp.List[tp.Union[pd.DataFrame, str, Path]], **process_kwargs
    ) -> tp.List[pd.DataFrame]:
        """Process multiple datasets."""
        return [self.process_dataframe(data, **process_kwargs) for data in datasets]
