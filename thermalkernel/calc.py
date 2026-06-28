"""
Чистые функции расчёта тепловой защиты по СП 50.13330.2012.
Без сети, без LLM, без I/O. Детерминированные.
"""
from __future__ import annotations
from thermalkernel import refs
from thermalkernel.data import get_r_norm_coeffs
from thermalkernel.models import (
    CalcResult,
    Climate,
    Construction,
    Layer,
    Material,
    OperationCondition,
)

# список ссылок на СП для каждого шага расчёта (порядок соответствует CalcResult.sp_refs)
_SP_REFS = [
    refs.R_LAYER,
    refs.ALPHA_IN,
    refs.ALPHA_OUT,
    refs.R0_USL,
    refs.R0_PR,
    refs.GSOP,
    refs.R_NORM,
    refs.VERDICT,
]


def r_layer(layer: Layer, material: Material, condition: OperationCondition) -> float:
    """
    Термическое сопротивление одного слоя, м²·°C/Вт.

    R = δ / λ
    Ref: СП 50.13330.2012, п. 8.4, формула (8.4).
    """
    lam = material.lambda_a if condition == OperationCondition.A else material.lambda_b
    return layer.thickness / lam


def r0_usl(
    construction: Construction,
    materials: dict[str, Material],
    alpha_in: float,
    alpha_out: float,
) -> tuple[float, list[float]]:
    """
    Условное сопротивление теплопередаче конструкции, м²·°C/Вт.

    R₀_усл = 1/αв + ΣR_слоёв + 1/αн
    Ref: СП 50.13330.2012, п. 8.1, формула (8.1).

    Возвращает (r0_usl, [r_layer_1, r_layer_2, ...]).
    """
    condition = construction.operation_condition
    r_layers: list[float] = []
    for layer in construction.layers:
        if layer.material not in materials:
            raise KeyError(
                f"Материал '{layer.material}' не найден в справочнике. "
                f"Доступные: {', '.join(materials.keys())}"
            )
        mat = materials[layer.material]
        r_layers.append(r_layer(layer, mat, condition))

    r_sum = sum(r_layers)
    result = 1.0 / alpha_in + r_sum + 1.0 / alpha_out
    return result, r_layers


def r0_pr(r0_usl_value: float, r_coef: float = 1.0) -> float:
    """
    Приведённое сопротивление теплопередаче, м²·°C/Вт.

    R₀_пр = R₀_усл · r
    r — коэффициент теплотехнической однородности.
    По умолчанию r=1.0 (мостики холода — отдельная задача, Этап 2).
    Ref: СП 50.13330.2012, прил. Е, п. Е.1.
    """
    # r=1.0: мостики холода в этом прогоне не учитываются (Этап 2)
    return r0_usl_value * r_coef


def gsop(t_v: float, climate: Climate) -> float:
    """
    Градусо-сутки отопительного периода, °C·сут.

    ГСОП = (tв − tот) · zот
    Ref: СП 50.13330.2012, п. 5.3, формула (5.2).
    """
    return (t_v - climate.t_ot) * climate.z_ot


def r_norm(construction_type: str, gsop_value: float, coeffs: dict) -> float:
    """
    Нормируемое (требуемое) сопротивление теплопередаче, м²·°C/Вт.

    Rнорм = a · ГСОП + b
    Ref: СП 50.13330.2012, п. 5.3, табл. 3.
    """
    a = coeffs["a"]
    b = coeffs["b"]
    return a * gsop_value + b


def _extra_insulation_mm(
    r_deficit: float,
    lambda_insulation: float,
) -> float:
    """
    Оценка минимальной толщины дополнительного утепления, мм.

    δ = R_дефицит · λ_утеплителя × 1000 (для перевода в мм).
    Рассчитывается для λ минваты (условие Б) как типового утеплителя.
    """
    return r_deficit * lambda_insulation * 1000.0


def evaluate(
    construction: Construction,
    climate: Climate,
    t_v: float,
    materials: dict[str, Material],
    coefficients: dict,
) -> CalcResult:
    """
    Полный расчёт тепловой защиты конструкции.

    Шаги:
    1. R_слоя для каждого слоя (δ/λ).
    2. R₀_усл = 1/αв + ΣR + 1/αн.
    3. R₀_пр = R₀_усл · r (r=1.0, мостики не учтены).
    4. ГСОП = (tв − tот)·zот.
    5. Rнорм = a·ГСОП + b.
    6. Вердикт: R₀_пр ≥ Rнорм → «соответствует», иначе «не соответствует».
    7. При несоответствии: оценка доп. утепления минватой (λ=0.06 Вт/(м·°C), усл. Б).

    Ref: СП 50.13330.2012, пп. 5.3, 8.1, 8.4, прил. Е; СП 131.13330.2020.
    """
    alpha_in = coefficients["surface_heat_transfer"]["alpha_in"]["value"]
    alpha_out = coefficients["surface_heat_transfer"]["alpha_out"]["value"]

    # шаг 1–2: условное сопротивление
    r0u, r_layers = r0_usl(construction, materials, alpha_in, alpha_out)

    # шаг 3: приведённое (мостики не учтены → r=1.0)
    r0p = r0_pr(r0u)

    # шаг 4: ГСОП
    gsop_val = gsop(t_v, climate)

    # шаг 5: нормируемое
    coeffs_rn = get_r_norm_coeffs(construction.type.value, coefficients)
    r_norm_val = r_norm(construction.type.value, gsop_val, coeffs_rn)

    # шаг 6: вердикт
    ok = r0p >= r_norm_val
    verdict = "соответствует" if ok else "не соответствует"

    # шаг 7: доп. утепление
    extra_mm: float | None = None
    if not ok:
        deficit = r_norm_val - r0p
        # используем λ минваты (условие Б) как типовой утеплитель для оценки
        lambda_minvata_b = 0.06
        extra_mm = _extra_insulation_mm(deficit, lambda_minvata_b)

    return CalcResult(
        r_layers=r_layers,
        r0_usl=r0u,
        r0_pr=r0p,
        gsop=gsop_val,
        r_norm=r_norm_val,
        verdict=verdict,
        required_extra_insulation_mm=extra_mm,
        sp_refs=list(_SP_REFS),
    )
