"""Budget arithmetic uses actual supplied prices, never market assumptions."""
from decimal import Decimal


def estimate(budget, spent, non_gpu, reserve, hourly_price, seconds_per_update, remaining_updates):
    values = [Decimal(str(x)) for x in (budget, spent, non_gpu, reserve, hourly_price, seconds_per_update, remaining_updates)]
    if not all(x.is_finite() and x >= 0 for x in values) or values[4] == 0:
        raise ValueError('finite nonnegative values and positive hourly price required')
    total, spent, other, reserve, rate, seconds, updates = values
    available = total - spent - other - reserve
    projected = seconds * updates / Decimal(3600) * rate
    return dict(available_gpu_hours=str(max(Decimal(0), available) / rate),
                projected_gpu_cost=str(projected), may_start=available >= projected)
