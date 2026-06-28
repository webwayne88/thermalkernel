from __future__ import annotations

from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, field_validator, model_validator


class ConstructionType(str, Enum):
    WALL = "wall"
    ROOF = "roof"
    FLOOR = "floor"
    WINDOW = "window"


class OperationCondition(str, Enum):
    A = "A"
    B = "B"


class Material(BaseModel):
    name: str
    lambda_a: float  # теплопроводность при условии эксплуатации А, Вт/(м·°C)
    lambda_b: float  # теплопроводность при условии эксплуатации Б, Вт/(м·°C)
    source: str      # ссылка на пункт/таблицу СП

    @field_validator("lambda_a", "lambda_b")
    @classmethod
    def lambda_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Теплопроводность должна быть > 0, получено: {v}")
        return v


class Layer(BaseModel):
    material: str   # ключ материала в справочнике
    thickness: float  # толщина слоя в метрах

    @field_validator("thickness")
    @classmethod
    def thickness_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Толщина слоя должна быть > 0 м, получено: {v}")
        return v


class Construction(BaseModel):
    type: ConstructionType
    layers: list[Layer]
    area: float                            # площадь конструкции, м²
    n: float = 1.0                         # коэффициент учёта положения наружной поверхности (СП 50, табл. 6)
    operation_condition: OperationCondition = OperationCondition.B  # условие эксплуатации А или Б

    @field_validator("area")
    @classmethod
    def area_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Площадь конструкции должна быть > 0 м², получено: {v}")
        return v

    @model_validator(mode="after")
    def layers_not_empty(self) -> "Construction":
        if not self.layers:
            raise ValueError("Конструкция должна содержать хотя бы один слой")
        return self


class Climate(BaseModel):
    city: str
    t5: float    # температура наиболее холодной пятидневки (обеспеченность 0.92), °C
    t_ot: float  # средняя температура отопительного периода, °C
    z_ot: float  # продолжительность отопительного периода, сут.
    source: str  # ссылка на СП 131


class CalcResult(BaseModel):
    r_layers: list[float]          # термическое сопротивление каждого слоя, м²·°C/Вт
    r0_usl: float                  # условное сопротивление теплопередаче, м²·°C/Вт
    r0_pr: float                   # приведённое сопротивление теплопередаче, м²·°C/Вт
    gsop: float                    # градусо-сутки отопительного периода, °C·сут
    r_norm: float                  # нормируемое (требуемое) сопротивление, м²·°C/Вт
    verdict: Literal["соответствует", "не соответствует"]
    required_extra_insulation_mm: Optional[float] = None  # доп. утепление при несоответствии, мм
    sp_refs: list[str]             # ссылки на пункты СП для каждого шага расчёта
