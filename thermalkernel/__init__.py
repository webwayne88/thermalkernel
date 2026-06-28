"""
ThermalKernel — детерминированное расчётное ядро тепловой защиты зданий.
Реализует расчёт по СП 50.13330.2012.

Публичный API:
  Этап 1 (конструкция):
    evaluate(construction, climate, t_v, materials, coefficients) -> CalcResult

  Этап 2 (здание):
    evaluate_building(building, climate, materials, coefficients) -> BuildingResult

  Модели:
    Material, Layer, Construction, ConstructionType, OperationCondition,
    Climate, CalcResult,
    Building, BuildingType, BuildingResult, ConstructionHeatLoss
"""
from thermalkernel.calc import evaluate, evaluate_building
from thermalkernel.models import (
    Building,
    BuildingResult,
    BuildingType,
    CalcResult,
    Climate,
    Construction,
    ConstructionHeatLoss,
    ConstructionType,
    Layer,
    Material,
    OperationCondition,
)

__all__ = [
    # функции
    "evaluate",
    "evaluate_building",
    # модели Этапа 1
    "CalcResult",
    "Climate",
    "Construction",
    "ConstructionType",
    "Layer",
    "Material",
    "OperationCondition",
    # модели Этапа 2
    "Building",
    "BuildingResult",
    "BuildingType",
    "ConstructionHeatLoss",
]

__version__ = "0.2.0"
