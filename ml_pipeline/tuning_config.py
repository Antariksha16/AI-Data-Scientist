from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class TuningConfig:
    """Bounded configuration for training-set hyperparameter search."""

    n_iter: int = 12
    top_models: int = 2
    random_state: int = 42
    n_jobs: int = 1

    def validate(self) -> None:
        if self.n_iter < 1:
            raise ValueError("n_iter must be at least 1.")

        if self.top_models < 1:
            raise ValueError("top_models must be at least 1.")

        if self.n_jobs == 0 or self.n_jobs < -1:
            raise ValueError(
                "n_jobs must be -1 or a positive integer."
            )

    def to_dict(self) -> dict:
        self.validate()
        return asdict(self)
