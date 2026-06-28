import argparse
import sys

from thermalkernel.calc import evaluate
from thermalkernel.data import get_climate, get_material, load_climate, load_coefficients, load_materials
from thermalkernel.models import Construction, ConstructionType, Layer, OperationCondition


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


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.list_materials:
        print_materials()
        return

    if args.list_cities:
        print_cities()
        return

    print_report(args)


if __name__ == "__main__":
    main()
