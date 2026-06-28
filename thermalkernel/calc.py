"""
Чистые функции расчёта тепловой защиты по СП 50.13330.2012.
Без сети, без LLM, без I/O. Детерминированные.
"""
from __future__ import annotations
from thermalkernel import refs
from thermalkernel.data import get_alpha, get_r_norm_coeffs, load_energy_classes, get_normative_heat_demand
from thermalkernel.models import (
    Building,
    BuildingResult,
    CalcResult,
    Climate,
    Construction,
    ConstructionHeatLoss,
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


def r_norm(gsop_value: float, coeffs: dict) -> float:
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


def q_transmission_single(
    construction: Construction,
    climate: Climate,
    t_v: float,
    materials: dict[str, Material],
    alpha_in: float,
    alpha_out: float,
) -> tuple[float, float]:
    """
    Трансмиссионные теплопотери одной ограждающей конструкции, Вт.

    Q_тр_i = A_i · (tв − tн) · n_i / R₀_пр_i

    tн — температура наиболее холодной пятидневки (t5) из климатических данных.

    Ref: СП 50.13330.2012, п. 9.1; СП 60.13330.2020, п. 6.3 (TODO: сверить пункты).

    Возвращает (q_w, r0_pr): теплопотери в Вт и приведённое сопротивление конструкции.
    """
    r0u, _ = r0_usl(construction, materials, alpha_in, alpha_out)
    r0p = r0_pr(r0u)
    # tн = температура наиболее холодной пятидневки
    delta_t = t_v - climate.t5
    q_w = construction.area * delta_t * construction.n / r0p
    return q_w, r0p


def q_infiltration(
    building: Building,
    climate: Climate,
    coefficients: dict,
) -> float:
    """
    Инфильтрационные теплопотери здания, Вт.

    Принята упрощённая модель:
        Q_инф = c_в · ρ_в · V · n_ин · (tв − tн) / 3600

    где:
        c_в   — удельная теплоёмкость воздуха, кДж/(кг·°C)
        ρ_в   — плотность воздуха, кг/м³
        V     — отапливаемый объём здания, м³
        n_ин  — кратность инфильтрации, 1/ч
        tв    — расчётная температура внутреннего воздуха, °C
        tн    — температура наиболее холодной пятидневки (t5), °C

    Результат — Вт. c_в задан в кДж/(кг·°C), поэтому перевод единиц: × 1000 (кДж→Дж) / 3600 (ч→с) = Дж/с = Вт.

    TODO: уточнить по СП 60.13330.2020, п. 8. Значения n_ин и параметры воздуха предварительные.

    Ref: СП 60.13330.2020, п. 8 (TODO: сверить пункт).
    """
    inf_cfg = coefficients["infiltration"]
    c_air = inf_cfg["c_air"]["value"]       # кДж/(кг·°C)
    rho_air = inf_cfg["rho_air"]["value"]   # кг/м³
    n_inf = inf_cfg["n_infiltration"]["value"]  # 1/ч
    delta_t = building.t_v - climate.t5    # °C
    # Вт = кДж/(кг·°C) · кг/м³ · м³ · 1/ч · °C · (1000 Дж/кДж) / (3600 с/ч)
    q_w = c_air * rho_air * building.heated_volume * n_inf * delta_t * 1000.0 / 3600.0
    return q_w


def specific_heat_demand(
    q_total_w: float,
    building: Building,
    climate: Climate,
) -> float:
    """
    Удельный расход тепловой энергии на отопление, кВт·ч/(м²·год).

    Упрощённая формула:
        q_уд = Q_сум · z_от · 24 / (А_от · 1000)

    где:
        Q_сум — суммарные теплопотери здания в расчётных условиях, Вт
        z_от  — продолжительность отопительного периода, сут
        24    — число часов в сутках
        А_от  — отапливаемая площадь, м²
        1000  — перевод Вт → кВт

    Ref: СП 50.13330.2012, п. 10.1 (TODO: сверить номер пункта и формулу).
    """
    return q_total_w * climate.z_ot * 24.0 / (building.heated_area * 1000.0)


def energy_class(specific_demand: float, normative_demand: float) -> str:
    """
    Класс энергетической эффективности здания по отклонению от нормативного.

    delta = (q_уд − q_норм) / q_норм

    Пороги класса берутся из справочника energy_classes.json.

    Ref: СП 50.13330.2012, прил. Б, табл. Б.1 (TODO: сверить).
    """
    classes = load_energy_classes()
    delta = (specific_demand - normative_demand) / normative_demand
    for entry in classes:
        d_min = entry["delta_min"]
        d_max = entry["delta_max"]
        # delta_min=None означает нет нижней границы (самый высокий класс)
        # delta_max=None означает нет верхней границы (самый низкий класс)
        below_max = (d_max is None) or (delta < d_max)
        above_min = (d_min is None) or (delta >= d_min)
        if above_min and below_max:
            return entry["class"]
    # Защитный fallback — не должен достигаться при корректном справочнике
    return classes[-1]["class"]


def evaluate_building(
    building: Building,
    climate: Climate,
    materials: dict[str, Material],
    coefficients: dict,
) -> BuildingResult:
    """
    Полный расчёт теплопотерь и класса энергоэффективности здания.

    Шаги:
    1. Трансмиссионные теплопотери для каждой ограждающей конструкции.
    2. Инфильтрационные теплопотери.
    3. Суммарные теплопотери.
    4. Удельный расход тепловой энергии на отопление, кВт·ч/(м²·год).
    5. Базовый (нормируемый) удельный расход по типу и этажности.
    6. Класс энергоэффективности по отклонению от базового.

    Ref: СП 50.13330.2012, пп. 9.1, 10.1, прил. Б; СП 60.13330.2020, п. 8.
    """
    per_construction: list[ConstructionHeatLoss] = []
    q_tr_total = 0.0

    alpha_in, alpha_out = get_alpha(coefficients)

    for constr in building.constructions:
        q_w, r0p = q_transmission_single(constr, climate, building.t_v, materials, alpha_in, alpha_out)
        q_tr_total += q_w
        per_construction.append(ConstructionHeatLoss(
            construction_type=constr.type.value,
            area=constr.area,
            r0_pr=r0p,
            heat_loss_w=q_w,
        ))

    q_inf = q_infiltration(building, climate, coefficients)
    q_total = q_tr_total + q_inf

    q_sp = specific_heat_demand(q_total, building, climate)

    norm_demand = get_normative_heat_demand(
        building.building_type.value, building.floors, coefficients
    )

    en_class = energy_class(q_sp, norm_demand)

    delta = (q_sp - norm_demand) / norm_demand

    sp_refs = [
        refs.Q_TRANSMISSION,
        refs.Q_INFILTRATION,
        refs.SPECIFIC_HEAT_DEMAND,
        refs.NORMATIVE_HEAT_DEMAND,
        refs.ENERGY_CLASS,
    ]

    return BuildingResult(
        q_transmission_w=q_tr_total,
        q_infiltration_w=q_inf,
        q_total_w=q_total,
        specific_heat_demand=q_sp,
        normative_heat_demand=norm_demand,
        delta_from_norm=delta,
        energy_class=en_class,
        per_construction=per_construction,
        sp_refs=sp_refs,
    )


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
    r_norm_val = r_norm(gsop_val, coeffs_rn)

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
