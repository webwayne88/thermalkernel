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


def get_material(key: str, materials: dict[str, Material] | None = None) -> Material:
    """Возвращает материал по ключу. Неизвестный ключ — явная ошибка."""
    if materials is None:
        materials = load_materials()
    if key not in materials:
        available = ", ".join(materials.keys())
        raise KeyError(
            f"Материал '{key}' не найден в справочнике. Доступные: {available}"
        )
    return materials[key]


def get_climate(city_key: str, climates: dict[str, Climate] | None = None) -> Climate:
    """Возвращает климатические данные по ключу города. Неизвестный город — явная ошибка."""
    if climates is None:
        climates = load_climate()
    if city_key not in climates:
        available = ", ".join(climates.keys())
        raise KeyError(
            f"Город '{city_key}' не найден в справочнике. Доступные: {available}"
        )
    return climates[city_key]


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


def get_alpha(coefficients: dict | None = None) -> tuple[float, float]:
    """Возвращает (αв, αн) — коэффициенты теплоотдачи поверхностей."""
    if coefficients is None:
        coefficients = load_coefficients()
    alpha_in = coefficients["surface_heat_transfer"]["alpha_in"]["value"]
    alpha_out = coefficients["surface_heat_transfer"]["alpha_out"]["value"]
    return alpha_in, alpha_out
