import argparse
import json
import sys
from pathlib import Path

from thermalkernel.calc import evaluate, evaluate_building
from thermalkernel.data import get_climate, get_material, load_climate, load_coefficients, load_materials
from thermalkernel.models import Building, BuildingType, Construction, ConstructionType, Layer, OperationCondition


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="thermalkernel",
        description="Расчёт тепловой защиты ограждающей конструкции по СП 50.13330.2012",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Примеры:
  python -m thermalkernel
  python -m thermalkernel --material gazobeton --thickness 400 --city novosibirsk
  python -m thermalkernel --material kirpich_keramicheskiy --thickness 510 --city moscow --t-in 22
  python -m thermalkernel --list-materials
  python -m thermalkernel --list-cities

ВНИМАНИЕ: инструмент носит расчётно-вспомогательный характер и не заменяет
проектную документацию, проходящую государственную экспертизу.
        """,
    )
    parser.add_argument(
        "--material", "-m",
        default="gazobeton",
        help="Ключ материала из справочника (по умолчанию: gazobeton)",
    )
    parser.add_argument(
        "--thickness", "-d",
        type=float,
        default=400.0,
        help="Толщина слоя в мм (по умолчанию: 400)",
    )
    parser.add_argument(
        "--city", "-c",
        default="novosibirsk",
        help="Ключ города из справочника (по умолчанию: novosibirsk)",
    )
    parser.add_argument(
        "--type", "-t",
        default="wall",
        choices=["wall", "roof", "floor", "window"],
        help="Тип конструкции (по умолчанию: wall)",
    )
    parser.add_argument(
        "--t-in",
        type=float,
        default=20.0,
        help="Расчётная температура внутреннего воздуха tв, °C (по умолчанию: 20)",
    )
    parser.add_argument(
        "--condition",
        default="B",
        choices=["A", "B"],
        help="Условие эксплуатации: A или B (по умолчанию: B)",
    )
    parser.add_argument(
        "--area",
        type=float,
        default=1.0,
        help="Площадь конструкции, м² (по умолчанию: 1.0, для нормоотчёта не влияет)",
    )
    parser.add_argument(
        "--list-materials",
        action="store_true",
        help="Показать список доступных материалов и выйти",
    )
    parser.add_argument(
        "--list-cities",
        action="store_true",
        help="Показать список доступных городов и выйти",
    )
    parser.add_argument(
        "--building",
        metavar="FILE.json",
        help="Расчёт здания (класс энергоэффективности). Передать путь к JSON-описанию здания.",
    )
    parser.add_argument(
        "--demo-building",
        action="store_true",
        help="Запустить демо-расчёт здания (встроенный пример).",
    )
    return parser


def print_materials() -> None:
    materials = load_materials()
    print("Доступные материалы:")
    print(f"  {'Ключ':<30} {'Название'}")
    print(f"  {'-'*30} {'-'*50}")
    for key, mat in materials.items():
        print(f"  {key:<30} {mat.name}")


def print_cities() -> None:
    climates = load_climate()
    print("Доступные города:")
    print(f"  {'Ключ':<20} {'Город':<20} {'t5':>6} {'tот':>6} {'zот':>6}")
    print(f"  {'-'*20} {'-'*20} {'-'*6} {'-'*6} {'-'*6}")
    for key, cl in climates.items():
        print(f"  {key:<20} {cl.city:<20} {cl.t5:>6.1f} {cl.t_ot:>6.1f} {cl.z_ot:>6.0f}")


def print_report(args: argparse.Namespace) -> None:
    materials = load_materials()
    climates = load_climate()
    coefficients = load_coefficients()

    try:
        material = get_material(args.material, materials)
        climate = get_climate(args.city, climates)
    except KeyError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)

    thickness_m = args.thickness / 1000.0  # мм → м

    try:
        layer = Layer(material=args.material, thickness=thickness_m)
    except Exception as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        construction = Construction(
            type=ConstructionType(args.type),
            layers=[layer],
            area=args.area,
            operation_condition=OperationCondition(args.condition),
        )
    except Exception as e:
        print(f"Ошибка при создании конструкции: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        result = evaluate(construction, climate, args.t_in, materials, coefficients)
    except Exception as e:
        print(f"Ошибка расчёта: {e}", file=sys.stderr)
        sys.exit(1)

    # вывод нормоотчёта
    print()
    print("=" * 60)
    print("  НОРМООТЧЁТ. ТЕПЛОВАЯ ЗАЩИТА ОГРАЖДАЮЩЕЙ КОНСТРУКЦИИ")
    print("  СП 50.13330.2012")
    print("=" * 60)
    print()
    print("ИСХОДНЫЕ ДАННЫЕ")
    print(f"  Конструкция : {args.type}")
    print(f"  Слой        : {material.name}, δ = {args.thickness:.0f} мм")
    print(f"  Условие экспл.: {args.condition}")
    print(f"  Город       : {climate.city}")
    print(f"  tв          : {args.t_in:.1f} °C")
    print(f"  tот         : {climate.t_ot:.1f} °C  (отопит. период)")
    print(f"  zот         : {climate.z_ot:.0f} сут.")
    print()
    print("РЕЗУЛЬТАТЫ РАСЧЁТА")
    print()

    # слои
    for i, (layer_obj, r_val) in enumerate(zip(construction.layers, result.r_layers), 1):
        lam = material.lambda_a if args.condition == "A" else material.lambda_b
        print(f"  Слой {i}: R = δ/λ = {layer_obj.thickness:.3f} / {lam:.4f} = {r_val:.4f} м²·°C/Вт")
    print(f"    Ref: {result.sp_refs[0]}")
    print()

    print(f"  R₀_усл = 1/αв + ΣR + 1/αн")
    print(f"         = 1/{coefficients['surface_heat_transfer']['alpha_in']['value']:.1f} + "
          f"{sum(result.r_layers):.4f} + "
          f"1/{coefficients['surface_heat_transfer']['alpha_out']['value']:.1f}")
    print(f"         = {result.r0_usl:.4f} м²·°C/Вт")
    print(f"    Ref: {result.sp_refs[3]}")
    print()

    print(f"  R₀_пр = R₀_усл · r = {result.r0_usl:.4f} × 1.00 (мостики холода — Этап 2)")
    print(f"        = {result.r0_pr:.4f} м²·°C/Вт")
    print(f"    Ref: {result.sp_refs[4]}")
    print()

    print(f"  ГСОП   = (tв − tот) · zот = ({args.t_in:.1f} − ({climate.t_ot:.1f})) · {climate.z_ot:.0f}")
    print(f"         = {result.gsop:.1f} °C·сут")
    print(f"    Ref: {result.sp_refs[5]}")
    print()

    print(f"  Rнорм  = a·ГСОП + b")
    coeffs_rn = coefficients["r_norm_coefficients"][args.type]
    print(f"         = {coeffs_rn['a']} × {result.gsop:.1f} + {coeffs_rn['b']}")
    print(f"         = {result.r_norm:.4f} м²·°C/Вт")
    print(f"    Ref: {result.sp_refs[6]}")
    print()

    print("-" * 60)
    print(f"  Проверка: R₀_пр ({result.r0_pr:.4f}) >= Rнорм ({result.r_norm:.4f}) ?")
    verdict_upper = result.verdict.upper()
    print(f"  ВЕРДИКТ : {verdict_upper}")
    print(f"    Ref: {result.sp_refs[7]}")
    print("-" * 60)

    if result.required_extra_insulation_mm is not None:
        print()
        print(f"  Дефицит : {result.r_norm - result.r0_pr:.4f} м²·°C/Вт")
        print(f"  Дополнительное утепление (минвата, λ=0.06 Вт/(м·°C), усл.Б):")
        print(f"    ≈ {result.required_extra_insulation_mm:.0f} мм")

    print()
    print("ВНИМАНИЕ: расчёт носит вспомогательный характер и не заменяет")
    print("проектную документацию, проходящую государственную экспертизу.")
    print()


def _make_demo_building() -> Building:
    """
    Встроенный демо-пример здания для быстрой проверки.

    Трёхэтажный жилой дом в Москве:
    - стена:  кирпич 0.51 м + минвата 0.1 м, A=300 м²
    - крыша:  минвата 0.2 м, A=200 м²
    - окна:   однокамерный стеклопакет, A=60 м²
    - отапливаемая площадь: 600 м², объём: 1800 м³
    """
    constructions = [
        Construction(
            type=ConstructionType.WALL,
            layers=[
                Layer(material="kirpich_keramicheskiy", thickness=0.51),
                Layer(material="minvata", thickness=0.10),
            ],
            area=300.0,
            operation_condition=OperationCondition.B,
        ),
        Construction(
            type=ConstructionType.ROOF,
            layers=[
                Layer(material="minvata", thickness=0.20),
            ],
            area=200.0,
            operation_condition=OperationCondition.B,
        ),
        Construction(
            type=ConstructionType.WINDOW,
            layers=[
                Layer(material="steklo", thickness=0.006),
            ],
            area=60.0,
            operation_condition=OperationCondition.B,
        ),
    ]
    return Building(
        city="moscow",
        building_type=BuildingType.RESIDENTIAL,
        floors=3,
        t_v=20.0,
        constructions=constructions,
        heated_area=600.0,
        heated_volume=1800.0,
    )


def _load_building_from_json(path: str) -> Building:
    """Загружает описание здания из JSON-файла."""
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except FileNotFoundError:
        print(f"Ошибка: файл '{path}' не найден.", file=sys.stderr)
        sys.exit(1)
    except json.JSONDecodeError as e:
        print(f"Ошибка разбора JSON: {e}", file=sys.stderr)
        sys.exit(1)
    try:
        return Building(**raw)
    except Exception as e:
        print(f"Ошибка при создании модели здания: {e}", file=sys.stderr)
        sys.exit(1)


def print_building_report(building: Building) -> None:
    """Выводит отчёт по расчёту здания: теплопотери и класс энергоэффективности."""
    materials = load_materials()
    climates = load_climate()
    coefficients = load_coefficients()

    try:
        climate = get_climate(building.city, climates)
    except KeyError as e:
        print(f"Ошибка: {e}", file=sys.stderr)
        sys.exit(1)

    try:
        result = evaluate_building(building, climate, materials, coefficients)
    except Exception as e:
        print(f"Ошибка расчёта здания: {e}", file=sys.stderr)
        sys.exit(1)

    print()
    print("=" * 65)
    print("  РАСЧЁТ ЗДАНИЯ. ТЕПЛОПОТЕРИ И КЛАСС ЭНЕРГОЭФФЕКТИВНОСТИ")
    print("  СП 50.13330.2012 / СП 60.13330.2020")
    print("=" * 65)
    print()
    print("ИСХОДНЫЕ ДАННЫЕ")
    print(f"  Город                   : {climate.city}")
    print(f"  Тип здания              : {building.building_type.value}")
    print(f"  Этажей                  : {building.floors}")
    print(f"  tв                      : {building.t_v:.1f} °C")
    print(f"  tн (t5)                 : {climate.t5:.1f} °C")
    print(f"  Отопит. период          : {climate.z_ot:.0f} сут  (tот={climate.t_ot:.1f} °C)")
    print(f"  Отапливаемая площадь    : {building.heated_area:.1f} м²")
    print(f"  Отапливаемый объём      : {building.heated_volume:.1f} м³")
    print()
    print("ТРАНСМИССИОННЫЕ ТЕПЛОПОТЕРИ ПО КОНСТРУКЦИЯМ")
    print(f"  {'Тип':<10} {'Площадь, м²':>12} {'R₀_пр, м²·°C/Вт':>18} {'Q_тр, Вт':>10}")
    print(f"  {'-'*10} {'-'*12} {'-'*18} {'-'*10}")
    for item in result.per_construction:
        print(
            f"  {item.construction_type:<10} {item.area:>12.1f} "
            f"{item.r0_pr:>18.3f} {item.heat_loss_w:>10.1f}"
        )
    print(f"  {'ИТОГО':>43} {result.q_transmission_w:>10.1f}")
    print(f"    Ref: {result.sp_refs[0]}")
    print()
    print("ИНФИЛЬТРАЦИОННЫЕ ТЕПЛОПОТЕРИ")
    print(f"  Q_инф = {result.q_infiltration_w:.1f} Вт")
    print(f"    Ref: {result.sp_refs[1]}")
    print()
    print("-" * 65)
    print(f"  Q_суммарные             = {result.q_total_w:.1f} Вт")
    print()
    print("УДЕЛЬНЫЙ РАСХОД ТЕПЛОВОЙ ЭНЕРГИИ НА ОТОПЛЕНИЕ")
    print(f"  q_уд                    = {result.specific_heat_demand:.1f} кВт·ч/(м²·год)")
    print(f"  q_норм (базовое)        = {result.normative_heat_demand:.1f} кВт·ч/(м²·год)")
    delta_pct = result.delta_from_norm * 100.0
    sign = "+" if delta_pct >= 0 else ""
    print(f"  Отклонение от нормы     = {sign}{delta_pct:.1f}%")
    print(f"    Ref: {result.sp_refs[2]}")
    print(f"    Ref: {result.sp_refs[3]}")
    print()
    print("-" * 65)
    print(f"  КЛАСС ЭНЕРГОЭФФЕКТИВНОСТИ : {result.energy_class}")
    print(f"    Ref: {result.sp_refs[4]}")
    print("-" * 65)
    print()
    print("ВНИМАНИЕ: расчёт носит вспомогательный характер. Коэффициенты")
    print("инфильтрации и пороги классов требуют сверки с актуальной редакцией СП.")
    print()


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_materials:
        print_materials()
        return

    if args.list_cities:
        print_cities()
        return

    if args.demo_building:
        building = _make_demo_building()
        print_building_report(building)
        return

    if args.building:
        building = _load_building_from_json(args.building)
        print_building_report(building)
        return

    print_report(args)


if __name__ == "__main__":
    main()
