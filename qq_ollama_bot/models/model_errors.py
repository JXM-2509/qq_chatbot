class ModelError(RuntimeError):
    """A model request failed; messages contain no provider response body."""


class ModelBusyError(ModelError):
    pass


class ModelTimeoutError(ModelError):
    pass


class ModelContextError(ModelError):
    pass


class ModelAuthError(ModelError):
    pass


class ModelBalanceError(ModelError):
    pass


class ModelRateLimitError(ModelError):
    pass


def model_error_message(error: ModelError) -> str:
    if isinstance(error, ModelAuthError):
        return "模型服务鉴权失败，请联系管理员检查 API Key。"
    if isinstance(error, ModelBalanceError):
        return "模型服务余额不足，请联系管理员充值。"
    if isinstance(error, ModelRateLimitError):
        return "模型服务请求受限，请稍后重试。"
    return "模型服务暂时不可用，请稍后重试。"
