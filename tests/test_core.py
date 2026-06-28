"""
Тесты ядра ThermalKernel (calc.py, models.py, data.py).

ВАЖНО: Все тестовые кейсы являются ручными поверками самосогласованности —
ожидаемые значения рассчитаны вручную по тем же формулам, что и само ядро.
Эти кейсы ТРЕБУЮТ замены на официальные примеры расчётов из СП 23-101 /
СП 50.13330.2012 и верификации независимым экспертом-теплотехником перед
использованием в нормативных целях.
"""

import pytest
from pydantic import ValidationError

from thermalkernel.calc import r_layer, r0_usl, gsop, r_norm, evaluate
from thermalkernel.data import (
    load_materials,
    load_climate,
    load_coefficients,
    get_material,
    get_climate,
    get_r_norm_coeffs,
)
from thermalkernel.models import (
    Climate,
    Construction,
    ConstructionType,
    Layer,
    Material,
    OperationCondition,
)


# ---------------------------------------------------------------------------
# Фикстуры
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def materials():
    return load_materials()


@pytest.fixture(scope="session")
def climates():
    return load_climate()


@pytest.fixture(scope="session")
def coefficients():
    return load_coefficients()


def _make_material(lam_a: float, lam_b: float) -> Material:
    return Material(name="test", lambda_a=lam_a, lambda_b=lam_b, source="test")


def _make_construction(
    layers: list[Layer],
    ctype: ConstructionType = ConstructionType.WALL,
    cond: OperationCondition = OperationCondition.B,
) -> Construction:
    return Construction(type=ctype, layers=layers, area=10.0, operation_condition=cond)


# ---------------------------------------------------------------------------
# 1. r_layer: δ/λ
# ---------------------------------------------------------------------------

class TestRLayer:
    def test_known_values_condition_b(self):
        """0.4 / 0.12 = 3.333... — ручная поверка."""
        layer = Layer(material="m", thickness=0.4)
        mat = _make_material(0.10, 0.12)
        result = r_layer(layer, mat, OperationCondition.B)
        assert pytest.approx(result, rel=1e-6) == 0.4 / 0.12

    def test_known_values_condition_a(self):
        """При условии А берётся lambda_a."""
        layer = Layer(material="m", thickness=0.4)
        mat = _make_material(0.10, 0.12)
        result = r_layer(layer, mat, OperationCondition.A)
        assert pytest.approx(result, rel=1e-6) == 0.4 / 0.10

    def test_thin_layer(self):
        """0.01 / 0.5 = 0.02."""
        layer = Layer(material="m", thickness=0.01)
        mat = _make_material(0.5, 0.5)
        result = r_layer(layer, mat, OperationCondition.B)
        assert pytest.approx(result, rel=1e-6) == 0.02

    def test_gazobeton_from_data(self, materials):
        """Газобетон из справочника: 0.4 / lambda_b."""
        mat = materials["gazobeton"]
        layer = Layer(material="gazobeton", thickness=0.4)
        expected = 0.4 / mat.lambda_b
        result = r_layer(layer, mat, OperationCondition.B)
        assert pytest.approx(result, rel=1e-6) == expected


# ---------------------------------------------------------------------------
# 2. r0_usl: 1/αв + ΣR + 1/αн
# ---------------------------------------------------------------------------

class TestR0Usl:
    def test_single_layer(self, materials):
        """
        Один слой газобетон 0.4 м, условие Б.
        R0_usl = 1/8.7 + 0.4/0.17 + 1/23
        Ручная поверка: ~2.5114
        """
        mat = materials["gazobeton"]
        # lambda_b газобетона = 0.17
        expected = 1 / 8.7 + 0.4 / mat.lambda_b + 1 / 23.0
        construction = _make_construction([Layer(material="gazobeton", thickness=0.4)])
        result, r_layers = r0_usl(construction, materials, alpha_in=8.7, alpha_out=23.0)
        assert pytest.approx(result, rel=1e-6) == expected
        assert len(r_layers) == 1
        assert pytest.approx(r_layers[0], rel=1e-6) == 0.4 / mat.lambda_b

    def test_two_layers(self, materials):
        """
        Два слоя: кирпич 0.25 м + минвата 0.1 м, условие Б.
        R0_usl = 1/8.7 + 0.25/0.81 + 0.1/0.06 + 1/23
        """
        mat_kir = materials["kirpich_keramicheskiy"]
        mat_min = materials["minvata"]
        r1 = 0.25 / mat_kir.lambda_b
        r2 = 0.1 / mat_min.lambda_b
        expected = 1 / 8.7 + r1 + r2 + 1 / 23.0
        construction = _make_construction([
            Layer(material="kirpich_keramicheskiy", thickness=0.25),
            Layer(material="minvata", thickness=0.1),
        ])
        result, r_layers = r0_usl(construction, materials, alpha_in=8.7, alpha_out=23.0)
        assert pytest.approx(result, rel=1e-6) == expected
        assert len(r_layers) == 2

    def test_unknown_material_raises(self, materials):
        """Неизвестный материал → KeyError с описанием."""
        construction = _make_construction([Layer(material="nonexistent_mat", thickness=0.3)])
        with pytest.raises(KeyError, match="nonexistent_mat"):
            r0_usl(construction, materials, alpha_in=8.7, alpha_out=23.0)


# ---------------------------------------------------------------------------
# 3. gsop: (t_в − t_от)·z_от
# ---------------------------------------------------------------------------

class TestGsop:
    def test_novosibirsk_t20(self, climates):
        """Новосибирск, t_в=20: ГСОП = (20 − (−8.7)) × 230 = 6601."""
        climate = climates["novosibirsk"]
        result = gsop(20.0, climate)
        expected = (20.0 - climate.t_ot) * climate.z_ot
        assert pytest.approx(result, rel=1e-9) == expected
        assert pytest.approx(result, abs=0.1) == 6601.0

    def test_moscow_t20(self, climates):
        """Москва, t_в=20: ГСОП = (20 − (−3.1)) × 214."""
        climate = climates["moscow"]
        result = gsop(20.0, climate)
        expected = (20.0 - climate.t_ot) * climate.z_ot
        assert pytest.approx(result, rel=1e-9) == expected

    def test_custom_climate(self):
        """Произвольный климат: прямая проверка формулы."""
        climate = Climate(city="test", t5=-30.0, t_ot=-10.0, z_ot=200.0, source="test")
        result = gsop(22.0, climate)
        assert pytest.approx(result, rel=1e-9) == (22.0 - (-10.0)) * 200.0


# ---------------------------------------------------------------------------
# 4. r_norm: a·ГСОП + b для каждого типа конструкции
# ---------------------------------------------------------------------------

class TestRNorm:
    def test_wall(self, coefficients):
        """wall: a=0.00035, b=1.4, ГСОП=6601 → 3.71035."""
        coeffs = get_r_norm_coeffs("wall", coefficients)
        result = r_norm("wall", 6601.0, coeffs)
        expected = 0.00035 * 6601.0 + 1.4
        assert pytest.approx(result, rel=1e-9) == expected
        assert pytest.approx(result, abs=0.001) == 3.71035

    def test_roof(self, coefficients):
        """roof: a=0.0005, b=2.2."""
        coeffs = get_r_norm_coeffs("roof", coefficients)
        result = r_norm("roof", 6601.0, coeffs)
        expected = 0.0005 * 6601.0 + 2.2
        assert pytest.approx(result, rel=1e-9) == expected

    def test_floor(self, coefficients):
        """floor: a=0.00045, b=1.9."""
        coeffs = get_r_norm_coeffs("floor", coefficients)
        result = r_norm("floor", 6601.0, coeffs)
        expected = 0.00045 * 6601.0 + 1.9
        assert pytest.approx(result, rel=1e-9) == expected

    def test_window(self, coefficients):
        """window: a=0.000075, b=0.15."""
        coeffs = get_r_norm_coeffs("window", coefficients)
        result = r_norm("window", 6601.0, coeffs)
        expected = 0.000075 * 6601.0 + 0.15
        assert pytest.approx(result, rel=1e-9) == expected

    def test_unknown_type_raises(self, coefficients):
        """Неизвестный тип → KeyError."""
        with pytest.raises(KeyError, match="unknown_type"):
            get_r_norm_coeffs("unknown_type", coefficients)


# ---------------------------------------------------------------------------
# 5. evaluate: полный сценарий — газобетон 400 мм, Новосибирск, t_в=20
# ---------------------------------------------------------------------------

class TestEvaluateFull:
    def test_gazobeton_400_novosibirsk_verdict(self, materials, climates, coefficients):
        """
        Газобетон 400 мм, Новосибирск, t_в=20°C.
        Ожидается: R0_пр ≈ 2.511, R_норм ≈ 3.710, вердикт «не соответствует».
        Ручная поверка (см. docstring модуля).
        """
        construction = _make_construction([Layer(material="gazobeton", thickness=0.4)])
        climate = climates["novosibirsk"]
        result = evaluate(construction, climate, 20.0, materials, coefficients)

        assert pytest.approx(result.r0_pr, abs=0.001) == 2.511
        assert pytest.approx(result.r_norm, abs=0.001) == 3.710
        assert result.verdict == "не соответствует"

    def test_sp_refs_not_empty(self, materials, climates, coefficients):
        """sp_refs должен содержать хотя бы одну ссылку на СП."""
        construction = _make_construction([Layer(material="gazobeton", thickness=0.4)])
        climate = climates["novosibirsk"]
        result = evaluate(construction, climate, 20.0, materials, coefficients)
        assert len(result.sp_refs) > 0
        # Все ссылки должны содержать «СП»
        for ref in result.sp_refs:
            assert "СП" in ref, f"Ссылка не содержит 'СП': {ref}"

    def test_extra_insulation_calculated_on_fail(self, materials, climates, coefficients):
        """При несоответствии required_extra_insulation_mm должен быть > 0."""
        construction = _make_construction([Layer(material="gazobeton", thickness=0.4)])
        climate = climates["novosibirsk"]
        result = evaluate(construction, climate, 20.0, materials, coefficients)
        assert result.required_extra_insulation_mm is not None
        assert result.required_extra_insulation_mm > 0

    def test_gsop_value_in_result(self, materials, climates, coefficients):
        """ГСОП в результате должен совпадать с ручным расчётом."""
        construction = _make_construction([Layer(material="gazobeton", thickness=0.4)])
        climate = climates["novosibirsk"]
        result = evaluate(construction, climate, 20.0, materials, coefficients)
        assert pytest.approx(result.gsop, abs=0.1) == 6601.0


# ---------------------------------------------------------------------------
# 6. Соответствие: толстая стена → «соответствует»
# ---------------------------------------------------------------------------

class TestEvaluateCompliant:
    def test_gazobeton_1200mm_novosibirsk_compliant(self, materials, climates, coefficients):
        """
        Газобетон 1200 мм, Новосибирск, t_в=20°C.
        R0_пр ≈ 7.22 >> R_норм ≈ 3.71 → вердикт «соответствует».
        Ручная поверка.
        """
        construction = _make_construction([Layer(material="gazobeton", thickness=1.2)])
        climate = climates["novosibirsk"]
        result = evaluate(construction, climate, 20.0, materials, coefficients)
        assert result.r0_pr >= result.r_norm
        assert result.verdict == "соответствует"

    def test_compliant_no_extra_insulation(self, materials, climates, coefficients):
        """При соответствии required_extra_insulation_mm должен быть None."""
        construction = _make_construction([Layer(material="gazobeton", thickness=1.2)])
        climate = climates["novosibirsk"]
        result = evaluate(construction, climate, 20.0, materials, coefficients)
        assert result.required_extra_insulation_mm is None

    def test_minvata_thick_wall_compliant(self, materials, climates, coefficients):
        """
        Кирпич 0.25 м + минвата 0.2 м, Москва, t_в=20°C.
        R0_пр должен быть больше R_норм.
        """
        mat_min = materials["minvata"]
        r_min = 0.2 / mat_min.lambda_b  # ≈ 3.33
        construction = _make_construction([
            Layer(material="kirpich_keramicheskiy", thickness=0.25),
            Layer(material="minvata", thickness=0.2),
        ])
        climate = climates["moscow"]
        result = evaluate(construction, climate, 20.0, materials, coefficients)
        assert result.r0_pr >= result.r_norm
        assert result.verdict == "соответствует"


# ---------------------------------------------------------------------------
# 7. Граничные и невалидные входные данные
# ---------------------------------------------------------------------------

class TestValidation:
    def test_layer_zero_thickness_raises(self):
        """δ = 0 → ValidationError."""
        with pytest.raises(ValidationError):
            Layer(material="m", thickness=0.0)

    def test_layer_negative_thickness_raises(self):
        """δ < 0 → ValidationError."""
        with pytest.raises(ValidationError):
            Layer(material="m", thickness=-0.1)

    def test_material_zero_lambda_raises(self):
        """λ = 0 → ValidationError."""
        with pytest.raises(ValidationError):
            Material(name="bad", lambda_a=0.0, lambda_b=0.5, source="test")

    def test_material_negative_lambda_raises(self):
        """λ < 0 → ValidationError."""
        with pytest.raises(ValidationError):
            Material(name="bad", lambda_a=-0.1, lambda_b=0.5, source="test")

    def test_construction_zero_area_raises(self):
        """area = 0 → ValidationError."""
        with pytest.raises(ValidationError):
            Construction(
                type=ConstructionType.WALL,
                layers=[Layer(material="m", thickness=0.3)],
                area=0.0,
            )

    def test_construction_negative_area_raises(self):
        """area < 0 → ValidationError."""
        with pytest.raises(ValidationError):
            Construction(
                type=ConstructionType.WALL,
                layers=[Layer(material="m", thickness=0.3)],
                area=-5.0,
            )

    def test_construction_empty_layers_raises(self):
        """Пустой список слоёв → ValidationError."""
        with pytest.raises(ValidationError):
            Construction(
                type=ConstructionType.WALL,
                layers=[],
                area=10.0,
            )

    def test_unknown_material_in_data_raises(self):
        """get_material с неизвестным ключом → KeyError."""
        materials = load_materials()
        with pytest.raises(KeyError, match="no_such_material"):
            get_material("no_such_material", materials)

    def test_unknown_city_raises(self):
        """get_climate с неизвестным городом → KeyError."""
        climates = load_climate()
        with pytest.raises(KeyError, match="atlantida"):
            get_climate("atlantida", climates)

    def test_unknown_construction_type_in_get_r_norm_raises(self):
        """get_r_norm_coeffs с неизвестным типом → KeyError (не дефолт)."""
        coefficients = load_coefficients()
        with pytest.raises(KeyError, match="fake_type"):
            get_r_norm_coeffs("fake_type", coefficients)


# ---------------------------------------------------------------------------
# 8. data.py: загрузка справочников
# ---------------------------------------------------------------------------

class TestDataLoaders:
    def test_load_materials_nonempty(self, materials):
        assert len(materials) > 0

    def test_load_climate_nonempty(self, climates):
        assert len(climates) > 0

    def test_load_coefficients_has_r_norm(self, coefficients):
        assert "r_norm_coefficients" in coefficients

    def test_known_materials_present(self, materials):
        for key in ("gazobeton", "minvata", "kirpich_keramicheskiy", "zhelezobeton", "penopolistirol"):
            assert key in materials, f"Материал {key} отсутствует в справочнике"

    def test_known_cities_present(self, climates):
        for key in ("novosibirsk", "moscow", "saint_petersburg", "yakutsk"):
            assert key in climates, f"Город {key} отсутствует в справочнике"
