from pydantic import BaseModel, Field


class TokenUsage(BaseModel):
    agent_name: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens


def track_token_usage(usage: TokenUsage) -> None:
    print(
        "token_usage "
        f"agent={usage.agent_name} "
        f"model={usage.model_name} "
        f"input_tokens={usage.input_tokens} "
        f"output_tokens={usage.output_tokens} "
        f"total_tokens={usage.total_tokens}"
    )
