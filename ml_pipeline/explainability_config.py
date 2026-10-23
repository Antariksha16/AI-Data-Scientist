from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class ExplainabilityConfig:
    """Validated controls for holdout model explanations."""

    n_repeats: int = 5
    random_state: int = 42
    max_rows: int = 2_000
    top_n: int = 20
    n_jobs: int = 1

    def validate(self) -> None:
        if not 1 <= self.n_repeats <= 50:
            raise ValueError(
                "n_repeats must be between 1 and 50."
            )

        if self.max_rows < 10:
            raise ValueError("max_rows must be at least 10.")

        if self.top_n < 1:
            raise ValueError("top_n must be at least 1.")

        if self.n_jobs == 0 or self.n_jobs < -1:
            raise ValueError(
                "n_jobs must be -1 or a positive integer."
            )

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)
