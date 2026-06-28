"""
Тесты Этапа 2 ядра ThermalKernel:
    q_transmission_single, q_infiltration, specific_heat_demand,
    energy_class, evaluate_building, валидация Building.

ВАЖНО: Все ожидаемые значения получены ручным расчётом по тем же формулам, что
использует ядро (самосогласованная поверка). Они НЕ являются официальными
эталонами СП. Перед использованием в нормативных целях каждый тест, помеченный
"# ТРЕБУЕТ эталона СП и верификации экспертом", должен быть заменён примером
расчёта из СП 23-101 / СП 50.13330 или проверен независимым теплотехником.
"""

import pytest
from pydantic import ValidationError

from thermalkernel.calc import (
    energy_class,
    evaluate_building,
    q_infiltration,
    q_transmission_single,
    specific_heat_demand,
)
from thermalkernel.data import load_climate, load_coefficients, load_materials
from thermalkernel.models import (
    Building,
    BuildingType,
    Climate,
    Construction,
    ConstructionHeatLoss,
    ConstructionType,
    Layer,
    OperationCondition,
)


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def materials():
    return load_materials()


@pytest.fixture(scope="session")
def coefficients():
    return load_coefficients()


@pytest.fixture(scope="session")
def climate_novosibirsk():
    return load_climate()["novosibirsk"]


def _wall_construction(thickness: float, area: float = 20.0) -> Construction:
    """Однослойная стена из газобетона."""
    return Construction(
        type=ConstructionType.WALL,
        layers=[Layer(material="gazobeton", thickness=thickness)],
        area=area,
        n=1.0,
        operation_condition=OperationCondition.B,
    )


def _demo_building(constructions=None) -> Building:
    """Демо-здание для интеграционных тестов."""
    if constructions is None:
        constructions = [_wall_construction(0.4, area=20.0)]
    return Building(
        city="novosibirsk",
        building_type=BuildingType.RESIDENTIAL,
        floors=5,
        t_v=20.0,
        constructions=constructions,
        heated_area=200.0,
        heated_volume=600.0,
    )


# ---------------------------------------------------------------------------
# 1. q_transmission_single: A·(tв−tн)·n / R₀_пр
# ---------------------------------------------------------------------------

class TestQTransmissionSingle:
    """
    # ТРЕБУЕТ эталона СП и верификации экспертом.
    Все expected-значения — ручная поверка по формуле Q = A·ΔT·n/R₀_пр.
    """

    def test_basic_known_values(self, materials, coefficients, climate_novosibirsk):
        """
        Газобетон 0.4 м, площадь 20 м², n=1.0, t_v=20, t5=-39 (Новосибирск).
        R₀_пр = 1/8.7 + 0.4/0.17 + 1/23 ≈ 2.5114 м²·°C/Вт
        Q = 20 · (20 − (−39)) · 1.0 / 2.5114 ≈ 469.86 Вт
        """
        construction = _wall_construction(0.4, area=20.0)
        alpha_in = coefficients["surface_heat_transfer"]["alpha_in"]["value"]
        alpha_out = coefficients["surface_heat_transfer"]["alpha_out"]["value"]
        q_w, r0p = q_transmission_single(
            construction, climate_novosibirsk, 20.0, materials, alpha_in, alpha_out
        )
        # Ручной расчёт R₀_пр
        expected_r0p = 1 / 8.7 + 0.4 / 0.17 + 1 / 23.0
        # Q = A · ΔT · n / R₀_пр
        expected_q = 20.0 * (20.0 - climate_novosibirsk.t5) * 1.0 / expected_r0p

        assert pytest.approx(r0p, rel=1e-6) == expected_r0p
        assert pytest.approx(q_w, rel=1e-6) == expected_q

    def test_larger_area_scales_linearly(self, materials, coefficients, climate_novosibirsk):
        """Удвоение площади → удвоение теплопотерь."""
        alpha_in = coefficients["surface_heat_transfer"]["alpha_in"]["value"]
        alpha_out = coefficients["surface_heat_transfer"]["alpha_out"]["value"]
        c10 = _wall_construction(0.4, area=10.0)
        c20 = _wall_construction(0.4, area=20.0)
        q10, _ = q_transmission_single(c10, climate_novosibirsk, 20.0, materials, alpha_in, alpha_out)
        q20, _ = q_transmission_single(c20, climate_novosibirsk, 20.0, materials, alpha_in, alpha_out)
        assert pytest.approx(q20, rel=1e-9) == 2 * q10

    def test_n_coefficient_scales_linearly(self, materials, coefficients, climate_novosibirsk):
        """n=0.5 даёт вдвое меньше теплопотерь, чем n=1.0."""
        c1 = Construction(
            type=ConstructionType.FLOOR,
            layers=[Layer(material="gazobeton", thickness=0.4)],
            area=20.0, n=1.0, operation_condition=OperationCondition.B,
        )
        c_half = Construction(
            type=ConstructionType.FLOOR,
            layers=[Layer(material="gazobeton", thickness=0.4)],
            area=20.0, n=0.5, operation_condition=OperationCondition.B,
        )
        alpha_in = coefficients["surface_heat_transfer"]["alpha_in"]["value"]
        alpha_out = coefficients["surface_heat_transfer"]["alpha_out"]["value"]
        q1, _ = q_transmission_single(c1, climate_novosibirsk, 20.0, materials, alpha_in, alpha_out)
        q_half, _ = q_transmission_single(c_half, climate_novosibirsk, 20.0, materials, alpha_in, alpha_out)
        assert pytest.approx(q_half, rel=1e-9) == 0.5 * q1

    def test_returns_positive_when_t_v_greater_than_t5(self, materials, coefficients, climate_novosibirsk):
        """t_v=20 > t5=-39: теплопотери положительные."""
        construction = _wall_construction(0.4)
        alpha_in = coefficients["surface_heat_transfer"]["alpha_in"]["value"]
        alpha_out = coefficients["surface_heat_transfer"]["alpha_out"]["value"]
        q_w, _ = q_transmission_single(
            construction, climate_novosibirsk, 20.0, materials, alpha_in, alpha_out
        )
        assert q_w > 0

    def test_r0pr_matches_manual(self, materials, coefficients, climate_novosibirsk):
        """r0p из q_transmission_single совпадает с ручным r0_pr(r0_usl(...))."""
        from thermalkernel.calc import r0_pr, r0_usl
        construction = _wall_construction(0.4)
        alpha_in = coefficients["surface_heat_transfer"]["alpha_in"]["value"]
        alpha_out = coefficients["surface_heat_transfer"]["alpha_out"]["value"]
        r0u, _ = r0_usl(construction, materials, alpha_in, alpha_out)
        expected_r0p = r0_pr(r0u)

        _, r0p_from_q = q_transmission_single(
            construction, climate_novosibirsk, 20.0, materials, alpha_in, alpha_out
        )
        assert pytest.approx(r0p_from_q, rel=1e-9) == expected_r0p


# ---------------------------------------------------------------------------
# 2. q_infiltration: c_в · ρ_в · V · n_ин · (tв − t5) · 1000 / 3600
# ---------------------------------------------------------------------------

class TestQInfiltration:
    """
    # ТРЕБУЕТ эталона СП и верификации экспертом.
    Ожидаемые значения — ручная поверка по формуле из docstring функции.
    """

    def test_known_values_novosibirsk(self, coefficients, climate_novosibirsk):
        """
        V=600 м³, t_v=20, t5=-39 (Новосибирск), жилое здание (residential, n=0.5).
        Q = 1.005 · 1.2 · 600 · 0.5 · (20−(−39)) · 1000 / 3600
          = 1.005 · 1.2 · 600 · 0.5 · 59 · 1000 / 3600
          ≈ 5929.5 Вт
        Ручная поверка (эталона СП нет).
        """
        building = _demo_building()
        q_inf = q_infiltration(building, climate_novosibirsk, coefficients)

        c_air = 1.005
        rho_air = 1.2
        n_inf = 0.5   # residential из справочника
        delta_t = 20.0 - climate_novosibirsk.t5
        expected = c_air * rho_air * 600.0 * n_inf * delta_t * 1000.0 / 3600.0

        assert pytest.approx(q_inf, rel=1e-6) == expected

    def test_residential_vs_nonresidential_n(self, coefficients, climate_novosibirsk):
        """
        Если для residential и nonresidential заданы одинаковые n=0.5 в справочнике,
        результат совпадает. Тест фиксирует, что диспетч по типу здания работает:
        функция не падает и возвращает > 0 для обоих типов.
        Ручная поверка (эталона СП нет).
        """
        b_res = Building(
            city="novosibirsk", building_type=BuildingType.RESIDENTIAL,
            floors=5, t_v=20.0, constructions=[_wall_construction(0.4)],
            heated_area=200.0, heated_volume=600.0,
        )
        b_non = Building(
            city="novosibirsk", building_type=BuildingType.NONRESIDENTIAL,
            floors=5, t_v=20.0, constructions=[_wall_construction(0.4)],
            heated_area=200.0, heated_volume=600.0,
        )
        q_res = q_infiltration(b_res, climate_novosibirsk, coefficients)
        q_non = q_infiltration(b_non, climate_novosibirsk, coefficients)
        # оба должны быть положительными
        assert q_res > 0
        assert q_non > 0
        # при одинаковом n (оба 0.5 в справочнике) значения равны
        n_res = coefficients["infiltration"]["n_infiltration"]["residential"]["value"]
        n_non = coefficients["infiltration"]["n_infiltration"]["nonresidential"]["value"]
        if n_res == n_non:
            assert pytest.approx(q_res, rel=1e-9) == q_non

    def test_volume_scales_linearly(self, coefficients, climate_novosibirsk):
        """Удвоение объёма → удвоение Q_инф. Ручная поверка."""
        b1 = Building(
            city="novosibirsk", building_type=BuildingType.RESIDENTIAL,
            floors=5, t_v=20.0, constructions=[_wall_construction(0.4)],
            heated_area=200.0, heated_volume=300.0,
        )
        b2 = Building(
            city="novosibirsk", building_type=BuildingType.RESIDENTIAL,
            floors=5, t_v=20.0, constructions=[_wall_construction(0.4)],
            heated_area=200.0, heated_volume=600.0,
        )
        q1 = q_infiltration(b1, climate_novosibirsk, coefficients)
        q2 = q_infiltration(b2, climate_novosibirsk, coefficients)
        assert pytest.approx(q2, rel=1e-9) == 2 * q1

    def test_positive_heat_loss(self, coefficients, climate_novosibirsk):
        """При tв > t5 инфильтрационные теплопотери > 0."""
        building = _demo_building()
        q_inf = q_infiltration(building, climate_novosibirsk, coefficients)
        assert q_inf > 0

    def test_formula_units_consistency(self, coefficients, climate_novosibirsk):
        """
        Проверка размерностной согласованности: результат в Вт.
        кДж/(кг·°C) · кг/м³ · м³ · 1/ч · °C · (1000 Дж/кДж) / (3600 с/ч) = Вт
        При V=3600 м³, ΔT=1°C, c=1, ρ=1, n=1 → Q = 1000 Вт.
        Ручная поверка (эталона СП нет).
        """
        b = Building(
            city="novosibirsk", building_type=BuildingType.RESIDENTIAL,
            floors=1, t_v=20.0, constructions=[_wall_construction(0.4)],
            heated_area=100.0, heated_volume=3600.0,
        )
        # n_infiltration задан как dict по типу здания — мокаем обе ветки
        custom_coef = {
            "infiltration": {
                "c_air": {"value": 1.0},
                "rho_air": {"value": 1.0},
                "n_infiltration": {
                    "residential": {"value": 1.0},
                    "nonresidential": {"value": 1.0},
                },
            }
        }
        # Используем climate с t5=19 → delta_t=1
        custom_climate = Climate(city="test", t5=19.0, t_ot=-10.0, z_ot=200.0, source="test")
        # V=3600, c=1, rho=1, n=1, delta_t=1 → Q = 1*1*3600*1*1*1000/3600 = 1000 Вт
        q_inf = q_infiltration(b, custom_climate, custom_coef)
        assert pytest.approx(q_inf, rel=1e-9) == 1000.0


# ---------------------------------------------------------------------------
# 3. specific_heat_demand: Q·z_от·24 / (A_от·1000)  → кВт·ч/(м²·год)
# ---------------------------------------------------------------------------

class TestSpecificHeatDemand:
    """
    # ТРЕБУЕТ эталона СП и верификации экспертом.
    Проверяется размерностная согласованность формулы на контрольных числах.

    specific_heat_demand возвращает кортеж (q_уд, q_losses_kwh, q_internal_gains_kwh, q_net_kwh).
    Без coefficients работает упрощённая формула: q_уд = Q·z·24/A/1000,
    q_gains=0, q_net=q_losses.
    """

    def test_known_values(self, climate_novosibirsk):
        """
        Q=1000 Вт, z_ot=200 сут, A_ot=100 м² (без coefficients — упрощённая формула).
        q_уд = 1000 · 200 · 24 / (100 · 1000) = 48.0 кВт·ч/(м²·год).
        Ручная поверка (эталона СП нет).
        """
        building = Building(
            city="novosibirsk", building_type=BuildingType.RESIDENTIAL,
            floors=5, t_v=20.0, constructions=[_wall_construction(0.4)],
            heated_area=100.0, heated_volume=300.0,
        )
        climate = Climate(city="test", t5=-39.0, t_ot=-10.0, z_ot=200.0, source="test")
        q_sp, q_losses, q_gains, q_net = specific_heat_demand(1000.0, building, climate)
        expected = 1000.0 * 200.0 * 24.0 / (100.0 * 1000.0)
        assert pytest.approx(q_sp, rel=1e-9) == expected
        assert pytest.approx(q_sp, rel=1e-9) == 48.0
        # без coefficients: нет бытовых поступлений
        assert q_gains == 0.0
        assert pytest.approx(q_net, rel=1e-9) == q_losses

    def test_unit_check_1w_1day_1sqm(self):
        """
        Q=1000 Вт, z_ot=1 сут, A_ot=24 м² → q_уд = 1000·1·24/(24·1000) = 1.0 кВт·ч/(м²·год).
        Проверка формулы: 1 кВт × 24 ч / 24 м² = 1 кВт·ч/м².
        Ручная поверка (эталона СП нет).
        """
        building = Building(
            city="novosibirsk", building_type=BuildingType.RESIDENTIAL,
            floors=1, t_v=20.0, constructions=[_wall_construction(0.4)],
            heated_area=24.0, heated_volume=60.0,
        )
        climate = Climate(city="test", t5=-39.0, t_ot=-10.0, z_ot=1.0, source="test")
        q_sp, _, _, _ = specific_heat_demand(1000.0, building, climate)
        assert pytest.approx(q_sp, rel=1e-9) == 1.0

    def test_scales_with_q_total(self, climate_novosibirsk):
        """
        Удвоение Q_total → удвоение q_уд (без coefficients, линейная зависимость).
        Ручная поверка (эталона СП нет).
        """
        building = _demo_building()
        q1, _, _, _ = specific_heat_demand(1000.0, building, climate_novosibirsk)
        q2, _, _, _ = specific_heat_demand(2000.0, building, climate_novosibirsk)
        assert pytest.approx(q2, rel=1e-9) == 2 * q1

    def test_scales_inversely_with_area(self, climate_novosibirsk):
        """
        Удвоение площади → вдвое меньше q_уд (без coefficients).
        Ручная поверка (эталона СП нет).
        """
        b_100 = Building(
            city="novosibirsk", building_type=BuildingType.RESIDENTIAL,
            floors=5, t_v=20.0, constructions=[_wall_construction(0.4)],
            heated_area=100.0, heated_volume=300.0,
        )
        b_200 = Building(
            city="novosibirsk", building_type=BuildingType.RESIDENTIAL,
            floors=5, t_v=20.0, constructions=[_wall_construction(0.4)],
            heated_area=200.0, heated_volume=600.0,
        )
        r_100, _, _, _ = specific_heat_demand(1000.0, b_100, climate_novosibirsk)
        r_200, _, _, _ = specific_heat_demand(1000.0, b_200, climate_novosibirsk)
        assert pytest.approx(r_200, rel=1e-9) == r_100 / 2

    def test_with_coefficients_gains_reduce_q_net(self, climate_novosibirsk, coefficients):
        """
        С coefficients: бытовые теплопоступления уменьшают q_net.
        q_net < q_losses → q_уд_полная < q_уд_упрощённая.
        Ручная поверка (эталона СП нет).
        """
        building = _demo_building()
        q_sp_simple, q_losses, q_gains_zero, _ = specific_heat_demand(1000.0, building, climate_novosibirsk)
        q_sp_full, q_losses2, q_gains, q_net = specific_heat_demand(1000.0, building, climate_novosibirsk, coefficients)
        # q_losses должны совпадать (одна формула)
        assert pytest.approx(q_losses, rel=1e-9) == q_losses2
        # бытовые поступления > 0
        assert q_gains > 0.0
        # нетто-потребность меньше потерь
        assert q_net < q_losses
        # удельный расход с полной формулой меньше упрощённой
        assert q_sp_full < q_sp_simple

    def test_with_coefficients_eta_sys_increases_q_sp(self, climate_novosibirsk, coefficients):
        """
        eta_sys < 1 увеличивает q_уд по сравнению с eta_sys=1.
        Мокаем coefficients с eta_sys=1.0 и сравниваем.
        Ручная поверка (эталона СП нет).
        """
        import copy
        building = _demo_building()
        coef_eta1 = copy.deepcopy(coefficients)
        coef_eta1["heating_system"]["eta_sys"]["value"] = 1.0

        q_sp_real, _, q_gains_real, q_net_real = specific_heat_demand(
            5000.0, building, climate_novosibirsk, coefficients
        )
        q_sp_eta1, _, _, q_net_eta1 = specific_heat_demand(
            5000.0, building, climate_novosibirsk, coef_eta1
        )
        # eta_sys < 1 → q_уд выше
        eta = coefficients["heating_system"]["eta_sys"]["value"]
        if eta < 1.0:
            assert q_sp_real > q_sp_eta1

    def test_tuple_traceability(self, climate_novosibirsk, coefficients):
        """
        Проверка согласованности кортежа (q_уд, q_losses, q_gains, q_net):
        q_net ≈ (q_losses − q_gains·v_r) / eta_sys,
        q_уд = q_net / A_от.
        Ручная поверка (эталона СП нет).
        """
        building = _demo_building()
        q_total = 5000.0
        q_sp, q_losses, q_gains, q_net = specific_heat_demand(
            q_total, building, climate_novosibirsk, coefficients
        )
        v_r = coefficients["heating_system"]["v_r"]["value"]
        eta_sys = coefficients["heating_system"]["eta_sys"]["value"]

        expected_q_losses = q_total * climate_novosibirsk.z_ot * 24.0 / 1000.0
        q_internal_per_sqm = coefficients["heat_gains"]["q_internal_per_sqm"]["residential"]["value"]
        expected_q_gains = q_internal_per_sqm * building.heated_area * climate_novosibirsk.z_ot * 24.0 / 1000.0
        expected_q_net = (expected_q_losses - expected_q_gains * v_r) / eta_sys
        expected_q_sp = expected_q_net / building.heated_area

        assert pytest.approx(q_losses, rel=1e-9) == expected_q_losses
        assert pytest.approx(q_gains, rel=1e-9) == expected_q_gains
        assert pytest.approx(q_net, rel=1e-9) == expected_q_net
        assert pytest.approx(q_sp, rel=1e-9) == expected_q_sp

    def test_q_net_floor_zero(self, coefficients):
        """
        Если бытовые поступления > теплопотерь, q_net не уходит в минус (защита floor(0)).
        Ручная поверка (эталона СП нет).
        """
        building = Building(
            city="novosibirsk", building_type=BuildingType.RESIDENTIAL,
            floors=5, t_v=20.0, constructions=[_wall_construction(0.4)],
            heated_area=10000.0, heated_volume=30000.0,
        )
        # Q_total крошечный, площадь огромная → поступления > потерь
        climate = Climate(city="test", t5=-1.0, t_ot=-0.5, z_ot=1.0, source="test")
        q_sp, _, _, q_net = specific_heat_demand(0.001, building, climate, coefficients)
        assert q_net >= 0.0
        assert q_sp >= 0.0


# ---------------------------------------------------------------------------
# 4. energy_class: пороги шкалы
# ---------------------------------------------------------------------------

class TestEnergyClass:
    """
    # ТРЕБУЕТ верификации экспертом: пороги классов взяты из energy_classes.json,
    который сам помечен TODO-сверкой с СП 50 / ПП РФ №18.
    """

    def test_a_plus_plus(self):
        """delta < -0.60 → A++. sp=30, norm=100 → delta=-0.70."""
        assert energy_class(30.0, 100.0) == "A++"

    def test_a_plus_plus_boundary(self):
        """delta = -0.61 (чуть ниже -0.60) → A++."""
        # sp = 100 * (1 - 0.61) = 39.0 → delta = (39-100)/100 = -0.61
        assert energy_class(39.0, 100.0) == "A++"

    def test_a_plus_at_lower_boundary(self):
        """delta = -0.60 точно → A+ (A+ включает delta_min=-0.60)."""
        # sp = 40.0 → delta = -0.60
        assert energy_class(40.0, 100.0) == "A+"

    def test_a_plus(self):
        """delta = -0.55 → A+."""
        # sp = 45.0 → delta = -0.55
        assert energy_class(45.0, 100.0) == "A+"

    def test_a(self):
        """delta = -0.45 → A."""
        # sp = 55.0 → delta = -0.45
        assert energy_class(55.0, 100.0) == "A"

    def test_c_normal(self):
        """delta = -0.15 → C (нормальный). sp=85, norm=100."""
        assert energy_class(85.0, 100.0) == "C"

    def test_c_upper_boundary(self):
        """delta < 0.0 → C. sp=99.99 → delta≈-0.0001 → C."""
        assert energy_class(99.99, 100.0) == "C"

    def test_d_at_zero(self):
        """delta = 0.0 → D (delta_min=0.0 включительно). sp=100, norm=100."""
        assert energy_class(100.0, 100.0) == "D"

    def test_d_inside(self):
        """delta = 0.10 → D."""
        # sp = 110.0 → delta = 0.10
        assert energy_class(110.0, 100.0) == "D"

    def test_e(self):
        """delta = 0.35 → E."""
        # sp = 135.0 → delta = 0.35
        assert energy_class(135.0, 100.0) == "E"

    def test_f(self):
        """delta = 0.60 → F. sp=160, norm=100."""
        assert energy_class(160.0, 100.0) == "F"

    def test_f_at_boundary(self):
        """delta = 0.50 точно → F (delta_min=0.50 включительно)."""
        # sp = 150.0 → delta = 0.50
        assert energy_class(150.0, 100.0) == "F"

    def test_result_is_string(self):
        """Возвращает строку."""
        result = energy_class(100.0, 100.0)
        assert isinstance(result, str)

    def test_all_known_norms_return_valid_class(self):
        """Для любого разумного sp функция возвращает непустую строку."""
        valid_classes = {"A++", "A+", "A", "B", "C", "D", "E", "F"}
        for sp in [10.0, 40.0, 55.0, 65.0, 75.0, 85.0, 100.0, 120.0, 140.0, 160.0]:
            result = energy_class(sp, 100.0)
            assert result in valid_classes, f"Неизвестный класс '{result}' для sp={sp}"


# ---------------------------------------------------------------------------
# 5. evaluate_building: демо-здание целиком
# ---------------------------------------------------------------------------

class TestEvaluateBuilding:
    """
    # ТРЕБУЕТ эталона СП и верификации экспертом.
    Ожидаемые значения — ручная поверка по формулам.
    """

    def test_q_total_equals_tr_plus_inf(self, materials, coefficients, climate_novosibirsk):
        """q_total_w = q_transmission_w + q_infiltration_w."""
        building = _demo_building()
        result = evaluate_building(building, climate_novosibirsk, materials, coefficients)
        assert pytest.approx(result.q_total_w, rel=1e-9) == (
            result.q_transmission_w + result.q_infiltration_w
        )

    def test_q_transmission_equals_sum_per_construction(
        self, materials, coefficients, climate_novosibirsk
    ):
        """Сумма per_construction.heat_loss_w == q_transmission_w."""
        # Два ограждения
        constructions = [
            _wall_construction(0.4, area=20.0),
            Construction(
                type=ConstructionType.ROOF,
                layers=[Layer(material="gazobeton", thickness=0.3)],
                area=50.0, n=1.0, operation_condition=OperationCondition.B,
            ),
        ]
        building = Building(
            city="novosibirsk", building_type=BuildingType.RESIDENTIAL,
            floors=5, t_v=20.0, constructions=constructions,
            heated_area=200.0, heated_volume=600.0,
        )
        result = evaluate_building(building, climate_novosibirsk, materials, coefficients)
        total_from_parts = sum(c.heat_loss_w for c in result.per_construction)
        assert pytest.approx(total_from_parts, rel=1e-9) == result.q_transmission_w

    def test_per_construction_count_matches_input(self, materials, coefficients, climate_novosibirsk):
        """Число записей per_construction = числу конструкций в здании."""
        constructions = [
            _wall_construction(0.4, area=20.0),
            Construction(
                type=ConstructionType.ROOF,
                layers=[Layer(material="minvata", thickness=0.2)],
                area=50.0, n=1.0, operation_condition=OperationCondition.B,
            ),
            Construction(
                type=ConstructionType.WINDOW,
                layers=[Layer(material="gazobeton", thickness=0.05)],
                area=5.0, n=1.0, operation_condition=OperationCondition.B,
            ),
        ]
        building = Building(
            city="novosibirsk", building_type=BuildingType.RESIDENTIAL,
            floors=5, t_v=20.0, constructions=constructions,
            heated_area=200.0, heated_volume=600.0,
        )
        result = evaluate_building(building, climate_novosibirsk, materials, coefficients)
        assert len(result.per_construction) == 3

    def test_per_construction_fields(self, materials, coefficients, climate_novosibirsk):
        """Каждая запись per_construction содержит корректные поля."""
        building = _demo_building()
        result = evaluate_building(building, climate_novosibirsk, materials, coefficients)
        for pc in result.per_construction:
            assert isinstance(pc, ConstructionHeatLoss)
            assert pc.area > 0
            assert pc.r0_pr > 0
            assert pc.heat_loss_w > 0
            assert pc.construction_type in ("wall", "roof", "floor", "window")

    def test_energy_class_filled(self, materials, coefficients, climate_novosibirsk):
        """energy_class в результате — непустая строка."""
        building = _demo_building()
        result = evaluate_building(building, climate_novosibirsk, materials, coefficients)
        assert isinstance(result.energy_class, str)
        assert len(result.energy_class) > 0

    def test_sp_refs_nonempty(self, materials, coefficients, climate_novosibirsk):
        """sp_refs содержит хотя бы одну ссылку, и каждая содержит 'СП'."""
        building = _demo_building()
        result = evaluate_building(building, climate_novosibirsk, materials, coefficients)
        assert len(result.sp_refs) > 0
        for ref in result.sp_refs:
            assert "СП" in ref, f"Ссылка не содержит 'СП': {ref}"

    def test_specific_heat_demand_value(self, materials, coefficients, climate_novosibirsk):
        """
        # ТРЕБУЕТ эталона СП и верификации экспертом.
        Демо-здание: одна стена газобетон 0.4 м / 20 м², V=600, A=200.
        Полная методика с вычетом бытовых поступлений и КПД системы.

        Ручная поверка:
            Q_тр = 20 · (20−(−39)) · 1.0 / (1/8.7 + 0.4/0.17 + 1/23) ≈ 469.86 Вт
            Q_инф = 1.005·1.2·600·0.5·59·1000/3600 ≈ 5929.5 Вт
            Q_total ≈ 6399.36 Вт
            Q_losses = 6399.36 · 230 · 24 / 1000 ≈ 35324.49 кВт·ч/год
            Q_gains = 17 · 200 · 230 · 24 / 1000 = 18768 кВт·ч/год
            Q_net = (35324.49 − 18768·0.8) / 0.85 ≈ 23894.23 кВт·ч/год
            q_уд = 23894.23 / 200 ≈ 119.47 кВт·ч/(м²·год)
        """
        building = _demo_building()
        result = evaluate_building(building, climate_novosibirsk, materials, coefficients)

        expected_q_tr = 20.0 * (20.0 - climate_novosibirsk.t5) * 1.0 / (
            1 / 8.7 + 0.4 / 0.17 + 1 / 23.0
        )
        c_air, rho_air, n_inf = 1.005, 1.2, 0.5
        expected_q_inf = c_air * rho_air * 600.0 * n_inf * (20.0 - climate_novosibirsk.t5) * 1000.0 / 3600.0
        expected_q_total = expected_q_tr + expected_q_inf
        z_ot = climate_novosibirsk.z_ot
        q_losses = expected_q_total * z_ot * 24.0 / 1000.0
        q_gains = 17.0 * 200.0 * z_ot * 24.0 / 1000.0
        v_r = coefficients["heating_system"]["v_r"]["value"]
        eta_sys = coefficients["heating_system"]["eta_sys"]["value"]
        q_net = (q_losses - q_gains * v_r) / eta_sys
        expected_q_sp = q_net / 200.0

        assert pytest.approx(result.specific_heat_demand, rel=1e-6) == expected_q_sp
        # трассируемость: поля BuildingResult согласованы
        assert pytest.approx(result.q_losses_kwh, rel=1e-6) == q_losses
        assert pytest.approx(result.q_internal_gains_kwh, rel=1e-6) == q_gains
        assert pytest.approx(result.q_net_kwh, rel=1e-6) == q_net

    def test_delta_from_norm_consistent(self, materials, coefficients, climate_novosibirsk):
        """delta_from_norm = (specific_heat_demand − normative_heat_demand) / normative_heat_demand."""
        building = _demo_building()
        result = evaluate_building(building, climate_novosibirsk, materials, coefficients)
        expected_delta = (
            result.specific_heat_demand - result.normative_heat_demand
        ) / result.normative_heat_demand
        assert pytest.approx(result.delta_from_norm, rel=1e-9) == expected_delta

    def test_normative_heat_demand_for_5floors(self, materials, coefficients, climate_novosibirsk):
        """Жилое, 5 этажей → norm_demand = 70.0 кВт·ч/(м²·год) из справочника."""
        building = _demo_building()
        result = evaluate_building(building, climate_novosibirsk, materials, coefficients)
        assert result.normative_heat_demand == 70.0

    def test_high_insulation_gives_better_class(self, materials, coefficients, climate_novosibirsk):
        """
        Хорошо утеплённое здание даёт лучший (меньший) класс, чем плохо утеплённое.
        Оба класса — из утверждённой шкалы.
        """
        valid_order = ["A++", "A+", "A", "B", "C", "D", "E", "F"]

        bad_building = Building(
            city="novosibirsk", building_type=BuildingType.RESIDENTIAL,
            floors=5, t_v=20.0,
            constructions=[_wall_construction(0.05, area=200.0)],  # тонкая стена
            heated_area=200.0, heated_volume=600.0,
        )
        good_building = Building(
            city="novosibirsk", building_type=BuildingType.RESIDENTIAL,
            floors=5, t_v=20.0,
            constructions=[_wall_construction(2.0, area=200.0)],  # толстая стена
            heated_area=200.0, heated_volume=600.0,
        )
        bad_result = evaluate_building(bad_building, climate_novosibirsk, materials, coefficients)
        good_result = evaluate_building(good_building, climate_novosibirsk, materials, coefficients)
        bad_idx = valid_order.index(bad_result.energy_class)
        good_idx = valid_order.index(good_result.energy_class)
        assert good_idx <= bad_idx


# ---------------------------------------------------------------------------
# 6. Валидация Building
# ---------------------------------------------------------------------------

class TestBuildingValidation:
    def test_heated_area_zero_raises(self):
        """heated_area = 0 → ValidationError."""
        with pytest.raises(ValidationError, match="Отапливаемая площадь"):
            Building(
                city="novosibirsk", building_type=BuildingType.RESIDENTIAL,
                floors=5, t_v=20.0,
                constructions=[_wall_construction(0.4)],
                heated_area=0.0, heated_volume=600.0,
            )

    def test_heated_area_negative_raises(self):
        """heated_area < 0 → ValidationError."""
        with pytest.raises(ValidationError):
            Building(
                city="novosibirsk", building_type=BuildingType.RESIDENTIAL,
                floors=5, t_v=20.0,
                constructions=[_wall_construction(0.4)],
                heated_area=-100.0, heated_volume=600.0,
            )

    def test_heated_volume_zero_raises(self):
        """heated_volume = 0 → ValidationError."""
        with pytest.raises(ValidationError, match="Отапливаемый объём"):
            Building(
                city="novosibirsk", building_type=BuildingType.RESIDENTIAL,
                floors=5, t_v=20.0,
                constructions=[_wall_construction(0.4)],
                heated_area=200.0, heated_volume=0.0,
            )

    def test_heated_volume_negative_raises(self):
        """heated_volume < 0 → ValidationError."""
        with pytest.raises(ValidationError):
            Building(
                city="novosibirsk", building_type=BuildingType.RESIDENTIAL,
                floors=5, t_v=20.0,
                constructions=[_wall_construction(0.4)],
                heated_area=200.0, heated_volume=-500.0,
            )

    def test_empty_constructions_raises(self):
        """Пустой список constructions → ValidationError."""
        with pytest.raises(ValidationError, match="Здание должно содержать"):
            Building(
                city="novosibirsk", building_type=BuildingType.RESIDENTIAL,
                floors=5, t_v=20.0,
                constructions=[],
                heated_area=200.0, heated_volume=600.0,
            )

    def test_floors_zero_raises(self):
        """floors = 0 → ValidationError."""
        with pytest.raises(ValidationError, match="Число этажей"):
            Building(
                city="novosibirsk", building_type=BuildingType.RESIDENTIAL,
                floors=0, t_v=20.0,
                constructions=[_wall_construction(0.4)],
                heated_area=200.0, heated_volume=600.0,
            )

    def test_floors_negative_raises(self):
        """floors < 0 → ValidationError."""
        with pytest.raises(ValidationError):
            Building(
                city="novosibirsk", building_type=BuildingType.RESIDENTIAL,
                floors=-1, t_v=20.0,
                constructions=[_wall_construction(0.4)],
                heated_area=200.0, heated_volume=600.0,
            )

    def test_valid_building_created_without_error(self):
        """Корректное здание создаётся без исключений."""
        b = _demo_building()
        assert b.heated_area == 200.0
        assert b.heated_volume == 600.0
        assert len(b.constructions) == 1


# ---------------------------------------------------------------------------
# 7. Мостики холода: r_coef < 1.0
# ---------------------------------------------------------------------------

class TestRCoef:
    """
    Тесты коэффициента теплотехнической однородности r (мостики холода).
    r_coef < 1.0 → R₀_пр уменьшается → теплопотери растут → вердикт может ухудшиться.
    Ручная поверка (эталона СП нет).
    """

    def test_r_coef_default_is_one(self):
        """По умолчанию r_coef=1.0 (однородная конструкция)."""
        c = Construction(
            type=ConstructionType.WALL,
            layers=[Layer(material="gazobeton", thickness=0.4)],
            area=10.0,
        )
        assert c.r_coef == 1.0

    def test_r_coef_zero_raises(self):
        """r_coef=0 → ValidationError (граница исключается: 0 < r ≤ 1)."""
        with pytest.raises(ValidationError, match="теплотехнической однородности"):
            Construction(
                type=ConstructionType.WALL,
                layers=[Layer(material="gazobeton", thickness=0.4)],
                area=10.0,
                r_coef=0.0,
            )

    def test_r_coef_negative_raises(self):
        """r_coef < 0 → ValidationError."""
        with pytest.raises(ValidationError):
            Construction(
                type=ConstructionType.WALL,
                layers=[Layer(material="gazobeton", thickness=0.4)],
                area=10.0,
                r_coef=-0.5,
            )

    def test_r_coef_greater_than_one_raises(self):
        """r_coef > 1 → ValidationError (физически невозможно)."""
        with pytest.raises(ValidationError):
            Construction(
                type=ConstructionType.WALL,
                layers=[Layer(material="gazobeton", thickness=0.4)],
                area=10.0,
                r_coef=1.01,
            )

    def test_r_coef_one_accepted(self):
        """r_coef=1.0 принимается (включительная верхняя граница)."""
        c = Construction(
            type=ConstructionType.WALL,
            layers=[Layer(material="gazobeton", thickness=0.4)],
            area=10.0,
            r_coef=1.0,
        )
        assert c.r_coef == 1.0

    def test_r_coef_08_accepted(self):
        """r_coef=0.8 принимается."""
        c = Construction(
            type=ConstructionType.WALL,
            layers=[Layer(material="gazobeton", thickness=0.4)],
            area=10.0,
            r_coef=0.8,
        )
        assert c.r_coef == 0.8

    def test_r_coef_reduces_r0_pr(self, materials, coefficients, climate_novosibirsk):
        """
        r_coef=0.8 → R₀_пр на 20% меньше, чем при r_coef=1.0.
        Ручная поверка (эталона СП нет).
        """
        from thermalkernel.calc import r0_pr, r0_usl
        c1 = Construction(
            type=ConstructionType.WALL,
            layers=[Layer(material="gazobeton", thickness=0.4)],
            area=10.0, r_coef=1.0,
        )
        c08 = Construction(
            type=ConstructionType.WALL,
            layers=[Layer(material="gazobeton", thickness=0.4)],
            area=10.0, r_coef=0.8,
        )
        alpha_in = coefficients["surface_heat_transfer"]["alpha_in"]["value"]
        alpha_out = coefficients["surface_heat_transfer"]["alpha_out"]["value"]
        r0u, _ = r0_usl(c1, materials, alpha_in, alpha_out)
        r0p_1 = r0_pr(r0u, 1.0)
        r0p_08 = r0_pr(r0u, 0.8)
        assert pytest.approx(r0p_08, rel=1e-9) == r0p_1 * 0.8

    def test_r_coef_increases_heat_loss(self, materials, coefficients, climate_novosibirsk):
        """
        r_coef=0.8 → теплопотери больше, чем при r_coef=1.0.
        Ручная поверка (эталона СП нет).
        """
        from thermalkernel.calc import q_transmission_single
        c1 = Construction(
            type=ConstructionType.WALL,
            layers=[Layer(material="gazobeton", thickness=0.4)],
            area=20.0, n=1.0, r_coef=1.0,
        )
        c08 = Construction(
            type=ConstructionType.WALL,
            layers=[Layer(material="gazobeton", thickness=0.4)],
            area=20.0, n=1.0, r_coef=0.8,
        )
        alpha_in = coefficients["surface_heat_transfer"]["alpha_in"]["value"]
        alpha_out = coefficients["surface_heat_transfer"]["alpha_out"]["value"]
        q1, r0p_1 = q_transmission_single(c1, climate_novosibirsk, 20.0, materials, alpha_in, alpha_out)
        q08, r0p_08 = q_transmission_single(c08, climate_novosibirsk, 20.0, materials, alpha_in, alpha_out)
        assert q08 > q1
        assert r0p_08 < r0p_1
        # пропорция: Q обратно пропорциональна R₀_пр → Q08/Q1 = R0p1/R0p08 = 1/0.8
        assert pytest.approx(q08 / q1, rel=1e-9) == 1.0 / 0.8

    def test_r_coef_can_change_verdict(self, materials, coefficients, climate_novosibirsk):
        """
        Конструкция с r_coef=1.0 может соответствовать норме,
        а с r_coef=0.8 — нет (мостики холода снижают R₀_пр).
        Ручная поверка: газобетон 0.4 м, Новосибирск — проверяем что r_coef влияет на вердикт.
        Если оба не соответствуют, проверяем хотя бы что r0_pr снижается.
        """
        from thermalkernel.calc import evaluate
        c1 = Construction(
            type=ConstructionType.WALL,
            layers=[Layer(material="gazobeton", thickness=0.4)],
            area=10.0, r_coef=1.0,
        )
        c08 = Construction(
            type=ConstructionType.WALL,
            layers=[Layer(material="gazobeton", thickness=0.4)],
            area=10.0, r_coef=0.8,
        )
        res1 = evaluate(c1, climate_novosibirsk, 20.0, materials, coefficients)
        res08 = evaluate(c08, climate_novosibirsk, 20.0, materials, coefficients)
        assert res08.r0_pr < res1.r0_pr
        # вердикт либо одинаково плохой, либо у r_coef=0.8 хуже
        verdict_order = {"соответствует": 0, "не соответствует": 1}
        assert verdict_order[res08.verdict] >= verdict_order[res1.verdict]

    def test_r_coef_propagates_through_evaluate_building(
        self, materials, coefficients, climate_novosibirsk
    ):
        """
        В evaluate_building per_construction.r0_pr отражает r_coef < 1.0.
        Ручная поверка (эталона СП нет).
        """
        c1 = Construction(
            type=ConstructionType.WALL,
            layers=[Layer(material="gazobeton", thickness=0.4)],
            area=20.0, n=1.0, r_coef=1.0,
        )
        c08 = Construction(
            type=ConstructionType.WALL,
            layers=[Layer(material="gazobeton", thickness=0.4)],
            area=20.0, n=1.0, r_coef=0.8,
        )
        b1 = Building(
            city="novosibirsk", building_type=BuildingType.RESIDENTIAL,
            floors=5, t_v=20.0, constructions=[c1],
            heated_area=200.0, heated_volume=600.0,
        )
        b08 = Building(
            city="novosibirsk", building_type=BuildingType.RESIDENTIAL,
            floors=5, t_v=20.0, constructions=[c08],
            heated_area=200.0, heated_volume=600.0,
        )
        res1 = evaluate_building(b1, climate_novosibirsk, materials, coefficients)
        res08 = evaluate_building(b08, climate_novosibirsk, materials, coefficients)
        # R₀_пр в per_construction снижен
        assert res08.per_construction[0].r0_pr < res1.per_construction[0].r0_pr
        # теплопотери выросли
        assert res08.q_transmission_w > res1.q_transmission_w
        assert res08.specific_heat_demand > res1.specific_heat_demand
