"""
ThermalKernel — детерминированное расчётное ядро тепловой защиты зданий.
Реализует расчёт по СП 50.13330.2012.

Публичный API:
  evaluate(construction, climate, t_v, materials, coefficients) -> CalcResult
  Модели: Material, Layer, Construction, ConstructionType, Climate, CalcResult, OperationCondition
"""
from thermalkernel.calc import evaluate
from thermalkernel.models import (
    CalcResult,
    Climate,
    Construction,
    ConstructionType,
    Layer,
    Material,
    OperationCondition,
)

__all__ = [
    "evaluate",
    "CalcResult",
    "Climate",
    "Construction",
    "ConstructionType",
    "Layer",
    "Material",
    "OperationCondition",
]

__version__ = "0.1.0"
