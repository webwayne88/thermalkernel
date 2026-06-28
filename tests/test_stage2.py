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
        V=600 м³, t_v=20, t5=-39 (Новосибирск).
        Q = 1.005 · 1.2 · 600 · 0.5 · (20−(−39)) · 1000 / 3600
          = 1.005 · 1.2 · 600 · 0.5 · 59 · 1000 / 3600
          ≈ 5929.5 Вт
        """
        building = _demo_building()
        q_inf = q_infiltration(building, climate_novosibirsk, coefficients)

        c_air = 1.005
        rho_air = 1.2
        n_inf = 0.5
        delta_t = 20.0 - climate_novosibirsk.t5
        expected = c_air * rho_air * 600.0 * n_inf * delta_t * 1000.0 / 3600.0

        assert pytest.approx(q_inf, rel=1e-6) == expected

    def test_volume_scales_linearly(self, coefficients, climate_novosibirsk):
        """Удвоение объёма → удвоение Q_инф."""
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
        """
        b = Building(
            city="novosibirsk", building_type=BuildingType.RESIDENTIAL,
            floors=1, t_v=20.0, constructions=[_wall_construction(0.4)],
            heated_area=100.0, heated_volume=3600.0,
        )
        # Подменяем коэффициенты inline через простой dict с нужными значениями
        custom_coef = {
            "infiltration": {
                "c_air": {"value": 1.0},
                "rho_air": {"value": 1.0},
                "n_infiltration": {"value": 1.0},
            }
        }
        # t5 Новосибирска -39, t_v=20 → delta_t=59 → не удобно
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
    """

    def test_known_values(self, climate_novosibirsk):
        """
        Q=1000 Вт, z_ot=200 сут, A_ot=100 м².
        q_уд = 1000 · 200 · 24 / (100 · 1000) = 48.0 кВт·ч/(м²·год)
        """
        building = Building(
            city="novosibirsk", building_type=BuildingType.RESIDENTIAL,
            floors=5, t_v=20.0, constructions=[_wall_construction(0.4)],
            heated_area=100.0, heated_volume=300.0,
        )
        climate = Climate(city="test", t5=-39.0, t_ot=-10.0, z_ot=200.0, source="test")
        result = specific_heat_demand(1000.0, building, climate)
        expected = 1000.0 * 200.0 * 24.0 / (100.0 * 1000.0)
        assert pytest.approx(result, rel=1e-9) == expected
        assert pytest.approx(result, rel=1e-9) == 48.0

    def test_unit_check_1w_1day_1sqm(self):
        """
        Q=1000 Вт, z_ot=1 сут, A_ot=24 м² → q_уд = 1000·1·24/(24·1000) = 1.0 кВт·ч/(м²·год).
        Проверка формулы: 1 кВт × 24 ч / 24 м² = 1 кВт·ч/м².
        """
        building = Building(
            city="novosibirsk", building_type=BuildingType.RESIDENTIAL,
            floors=1, t_v=20.0, constructions=[_wall_construction(0.4)],
            heated_area=24.0, heated_volume=60.0,
        )
        climate = Climate(city="test", t5=-39.0, t_ot=-10.0, z_ot=1.0, source="test")
        result = specific_heat_demand(1000.0, building, climate)
        assert pytest.approx(result, rel=1e-9) == 1.0

    def test_scales_with_q_total(self, climate_novosibirsk):
        """Удвоение Q_total → удвоение q_уд."""
        building = _demo_building()
        r1 = specific_heat_demand(1000.0, building, climate_novosibirsk)
        r2 = specific_heat_demand(2000.0, building, climate_novosibirsk)
        assert pytest.approx(r2, rel=1e-9) == 2 * r1

    def test_scales_inversely_with_area(self, climate_novosibirsk):
        """Удвоение площади → вдвое меньше q_уд."""
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
        r_100 = specific_heat_demand(1000.0, b_100, climate_novosibirsk)
        r_200 = specific_heat_demand(1000.0, b_200, climate_novosibirsk)
        assert pytest.approx(r_200, rel=1e-9) == r_100 / 2


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
        Ожидаемый q_уд ≈ 176.62 кВт·ч/(м²·год) — ручная поверка.
        """
        building = _demo_building()
        result = evaluate_building(building, climate_novosibirsk, materials, coefficients)
        expected_q_tr = 20.0 * (20.0 - climate_novosibirsk.t5) * 1.0 / (
            1 / 8.7 + 0.4 / 0.17 + 1 / 23.0
        )
        c_air, rho_air, n_inf = 1.005, 1.2, 0.5
        expected_q_inf = c_air * rho_air * 600.0 * n_inf * (20.0 - climate_novosibirsk.t5) * 1000.0 / 3600.0
        expected_q_total = expected_q_tr + expected_q_inf
        expected_q_sp = expected_q_total * climate_novosibirsk.z_ot * 24.0 / (200.0 * 1000.0)
        assert pytest.approx(result.specific_heat_demand, rel=1e-6) == expected_q_sp

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
