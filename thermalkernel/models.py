from __future__ import annotations

from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, field_validator, model_validator


class BuildingType(str, Enum):
    RESIDENTIAL = "residential"      # жилые многоквартирные
    NONRESIDENTIAL = "nonresidential"  # общественные/административные


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
    # Коэффициент теплотехнической однородности r (0 < r ≤ 1).
    # R₀_пр = R₀_усл · r  (СП 50.13330.2012, прил. Е, п. Е.1)
    # r = 1.0 означает однородную конструкцию без мостиков холода.
    # Значение r < 1.0 должен задать пользователь по результатам расчёта конструктива.
    r_coef: float = 1.0

    @field_validator("area")
    @classmethod
    def area_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Площадь конструкции должна быть > 0 м², получено: {v}")
        return v

    @field_validator("r_coef")
    @classmethod
    def r_coef_valid(cls, v: float) -> float:
        if not (0 < v <= 1.0):
            raise ValueError(f"Коэффициент теплотехнической однородности r должен быть в диапазоне (0; 1], получено: {v}")
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


class ConstructionHeatLoss(BaseModel):
    """Теплопотери одной ограждающей конструкции."""
    construction_type: str          # тип ограждения (wall, roof, …)
    area: float                     # площадь, м²
    r0_pr: float                    # приведённое сопротивление теплопередаче, м²·°C/Вт
    heat_loss_w: float              # трансмиссионные теплопотери, Вт


class Building(BaseModel):
    """Описание здания для расчёта теплопотерь и класса энергоэффективности."""
    city: str                              # ключ города из справочника
    building_type: BuildingType            # тип здания (жилое / нежилое)
    floors: int                            # количество этажей (для выбора базового удельного расхода)
    t_v: float                             # расчётная температура внутреннего воздуха, °C
    constructions: list[Construction]      # список ограждающих конструкций с площадями
    heated_area: float                     # отапливаемая площадь, м²
    heated_volume: float                   # отапливаемый объём здания, м³

    @field_validator("heated_area")
    @classmethod
    def heated_area_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Отапливаемая площадь должна быть > 0 м², получено: {v}")
        return v

    @field_validator("heated_volume")
    @classmethod
    def heated_volume_positive(cls, v: float) -> float:
        if v <= 0:
            raise ValueError(f"Отапливаемый объём должен быть > 0 м³, получено: {v}")
        return v

    @field_validator("floors")
    @classmethod
    def floors_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError(f"Число этажей должно быть > 0, получено: {v}")
        return v

    @model_validator(mode="after")
    def constructions_not_empty(self) -> "Building":
        if not self.constructions:
            raise ValueError("Здание должно содержать хотя бы одну ограждающую конструкцию")
        return self


class BuildingResult(BaseModel):
    """Результат расчёта теплопотерь и класса энергоэффективности здания."""
    # Трансмиссионные теплопотери, Вт (суммарные через все ограждения)
    q_transmission_w: float
    # Инфильтрационные теплопотери, Вт
    q_infiltration_w: float
    # Суммарные теплопотери здания (трансмиссия + инфильтрация), Вт
    q_total_w: float
    # Бытовые (внутренние) теплопоступления, кВт·ч/год
    q_internal_gains_kwh: float
    # Суммарные потери за отопительный период до вычета поступлений, кВт·ч/год
    q_losses_kwh: float
    # Суммарные потери за отопительный период после вычета бытовых поступлений, кВт·ч/год
    # q_net_kwh = (q_losses_kwh − q_internal_gains_kwh) / eta_sys
    q_net_kwh: float
    # Удельный расход тепловой энергии на отопление, кВт·ч/(м²·год)
    specific_heat_demand: float
    # Нормируемый (базовый) удельный расход тепловой энергии, кВт·ч/(м²·год)
    normative_heat_demand: float
    # Отклонение от нормативного значения (доля): (q_уд − q_норм) / q_норм
    delta_from_norm: float
    # Класс энергетической эффективности (A++, A+, A, B, C, D, E, F)
    energy_class: str
    # Разбивка теплопотерь по конструкциям
    per_construction: list[ConstructionHeatLoss]
    # Ссылки на пункты СП
    sp_refs: list[str]


class CalcResult(BaseModel):
    r_layers: list[float]          # термическое сопротивление каждого слоя, м²·°C/Вт
    r0_usl: float                  # условное сопротивление теплопередаче, м²·°C/Вт
    r0_pr: float                   # приведённое сопротивление теплопередаче, м²·°C/Вт
    gsop: float                    # градусо-сутки отопительного периода, °C·сут
    r_norm: float                  # нормируемое (требуемое) сопротивление, м²·°C/Вт
    verdict: Literal["соответствует", "не соответствует"]
    required_extra_insulation_mm: Optional[float] = None  # доп. утепление при несоответствии, мм
    sp_refs: list[str]             # ссылки на пункты СП для каждого шага расчёта
