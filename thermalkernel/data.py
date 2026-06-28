from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from thermalkernel.models import Material, Climate

_DATA_DIR = Path(__file__).parent / "data"


@lru_cache(maxsize=1)
def load_materials() -> dict[str, Material]:
    """Загружает справочник теплопроводности материалов из JSON."""
    path = _DATA_DIR / "materials.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, Material] = {}
    for key, entry in raw["materials"].items():
        result[key] = Material(
            name=entry["name"],
            lambda_a=entry["lambda_a"],
            lambda_b=entry["lambda_b"],
            source=entry["source"],
        )
    return result


@lru_cache(maxsize=1)
def load_climate() -> dict[str, Climate]:
    """Загружает климатические параметры городов из JSON."""
    path = _DATA_DIR / "climate.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    result: dict[str, Climate] = {}
    for key, entry in raw["cities"].items():
        result[key] = Climate(
            city=entry["city"],
            t5=entry["t5"],
            t_ot=entry["t_ot"],
            z_ot=entry["z_ot"],
            source=entry["source"],
        )
    return result


@lru_cache(maxsize=1)
def load_coefficients() -> dict:
    """Загружает коэффициенты a/b, αв, αн из JSON."""
    path = _DATA_DIR / "coefficients.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _lookup(mapping: dict, key: str, entity_name: str):
    """Ищет key в mapping, при отсутствии бросает KeyError с перечнем доступных."""
    if key not in mapping:
        available = ", ".join(mapping.keys())
        raise KeyError(
            f"{entity_name} '{key}' не найден в справочнике. Доступные: {available}"
        )
    return mapping[key]


def get_material(key: str, materials: dict[str, Material] | None = None) -> Material:
    """Возвращает материал по ключу. Неизвестный ключ — явная ошибка."""
    if materials is None:
        materials = load_materials()
    return _lookup(materials, key, "Материал")


def get_climate(city_key: str, climates: dict[str, Climate] | None = None) -> Climate:
    """Возвращает климатические данные по ключу города. Неизвестный город — явная ошибка."""
    if climates is None:
        climates = load_climate()
    return _lookup(climates, city_key, "Город")


def get_r_norm_coeffs(construction_type: str, coefficients: dict | None = None) -> dict:
    """Возвращает коэффициенты a и b для формулы Rнорм по типу конструкции."""
    if coefficients is None:
        coefficients = load_coefficients()
    r_norm = coefficients["r_norm_coefficients"]
    if construction_type not in r_norm:
        available = [k for k in r_norm.keys() if not k.startswith("_")]
        raise KeyError(
            f"Тип конструкции '{construction_type}' не найден. Доступные: {available}"
        )
    return r_norm[construction_type]


@lru_cache(maxsize=1)
def load_energy_classes() -> tuple[dict, ...]:
    """Загружает шкалу классов энергоэффективности из JSON."""
    path = _DATA_DIR / "energy_classes.json"
    raw = json.loads(path.read_text(encoding="utf-8"))
    return tuple(raw["classes"])


def get_normative_heat_demand(
    building_type: str,
    floors: int,
    coefficients: dict | None = None,
) -> float:
    """
    Возвращает базовый нормируемый удельный расход тепловой энергии, кВт·ч/(м²·год).

    Выбор по типу здания и числу этажей.
    Неизвестный тип → явная ошибка.
    """
    if coefficients is None:
        coefficients = load_coefficients()
    base = coefficients["specific_heat_demand_base"]
    if building_type not in base:
        available = [k for k in base.keys() if not k.startswith("_")]
        raise KeyError(
            f"Тип здания '{building_type}' не найден. Доступные: {available}"
        )
    by_floors = base[building_type]["by_floors"]
    if floors <= 3:
        return float(by_floors["3_or_less"])
    elif floors <= 9:
        return float(by_floors["4_to_9"])
    else:
        return float(by_floors["10_and_more"])


def get_alpha(coefficients: dict | None = None) -> tuple[float, float]:
    """Возвращает (αв, αн) — коэффициенты теплоотдачи поверхностей."""
    if coefficients is None:
        coefficients = load_coefficients()
    alpha_in = coefficients["surface_heat_transfer"]["alpha_in"]["value"]
    alpha_out = coefficients["surface_heat_transfer"]["alpha_out"]["value"]
    return alpha_in, alpha_out
