#!/usr/bin/env python3
"""归类应用命令行入口：日常上新、类目切换、在途采购单复查、人工复核安排。

在项目根目录运行，支持 Python 隔离模式（-I）：

    python3 -I apple_cli.py backtest
    python3 -I apple_cli.py classify --sku S006
    python3 -I apple_cli.py review-orders
    python3 -I apple_cli.py work-plan
    python3 -I apple_cli.py supplier --merchant M4
    python3 -I apple_cli.py trace --package P3
"""
import argparse
import json
import sys
from pathlib import Path

# 隔离模式（python -I）下 cwd 与 PYTHONPATH 均不生效，显式放入项目根。
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from apple_review.classifier import classify_with_human_override  # noqa: E402
from apple_review.commitment import review_all  # noqa: E402
from apple_review.evaluation import reliability_views, run_backtests  # noqa: E402
from apple_review.evidence import Packet  # noqa: E402
from apple_review.reporting import (  # noqa: E402
    classification_lines,
    operator_commitment_lines,
    supplier_lines,
    supplier_view,
)
from apple_review.taxonomy import build_taxonomies  # noqa: E402
from apple_review.workload import build_work_plan  # noqa: E402

DEFAULT_PACKET = PROJECT_ROOT / "资料" / "演练.json"


def load_context(path: Path):
    packet = Packet.load(path)
    taxonomies = build_taxonomies(packet.data["taxonomies"])
    return packet, taxonomies


def emit(payload, lines=None, as_json=False):
    if as_json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return
    for line in lines or []:
        print(line)


def cmd_classify(args):
    packet, taxonomies = load_context(args.packet)
    taxonomy = taxonomies[args.taxonomy]
    skus = [args.sku] if args.sku else sorted(packet.products)
    rows = [
        classify_with_human_override(
            packet, sku, taxonomy, method=args.method, as_of=args.as_of
        ).to_dict()
        for sku in skus
    ]
    emit(rows, classification_lines(rows), args.json)


def cmd_backtest(args):
    packet, taxonomies = load_context(args.packet)
    reports = run_backtests(packet, taxonomies)
    payload = [report.to_dict() for report in reports]
    if args.json:
        emit(payload, as_json=True)
        return
    print("== 同切分下两种做法对比（分组防泄漏，跨边整组剔除） ==")
    for report in reports:
        acc = "—" if report.accuracy is None else f"{report.accuracy:.3f}"
        soft = f"，软候选 {report.n_train_soft}" if report.n_train_soft else ""
        print(
            f"截止 {report.boundary} / {report.taxonomy} / {report.method}: "
            f"训练 {report.n_train}{soft} 考核 {report.n_test}（叶类 {report.n_leaf_test}，"
            f"未知 {report.n_unknown_test}），准确率 {acc}，弃权率 "
            f"{report.abstain_rate:.3f}，未知硬判 {report.forced_error}，"
            f"叶类误弃权 {report.abstain_wrong}，剔除跨边组 {report.n_excluded_groups}"
        )
        if report.note:
            print(f"   备注：{report.note}")
        for label, stats in report.per_label.items():
            rare = "（稀有类）" if stats["rare"] else ""
            print(
                f"   {label}{rare}: 支撑 {stats['support']} 召回 {stats['recall']:.3f}"
                f" 弃权 {stats['abstained']} 误判 {stats['misclassified_as']}"
            )
    print("\n== 可靠度视图 ==")
    views = reliability_views(reports)
    print("[稀有类别]")
    for row in views["rare_classes"]:
        print(
            f"  {row['boundary']} {row['taxonomy']} {row['method']} {row['label']}: "
            f"召回 {row['recall']:.3f}，支撑 {row['support']}，训练样本少"
        )
    print("[换版前后]")
    for key, rows in views["version_transition"].items():
        for row in rows:
            acc = "—" if row["accuracy"] is None else f"{row['accuracy']:.3f}"
            print(f"  {key} @ {row['boundary']}: 准确率 {acc}，训练标签 {row['labels_in_train']}")
    print("[未知类]")
    for row in views["unknown_class"]:
        print(
            f"  {row['boundary']} {row['taxonomy']} {row['method']}: "
            f"未知 {row['unknown_support']} 正确弃权 {row['abstain_correct']}，"
            f"硬判错误 {row['forced_error_on_unknown']}，叶类误弃权 {row['abstain_wrong_on_leaf']}"
        )


def cmd_review_orders(args):
    packet, taxonomies = load_context(args.packet)
    reviews = review_all(packet, taxonomies, as_of=args.as_of)
    payload = [review.to_dict() for review in reviews]
    emit(payload, operator_commitment_lines(reviews), args.json)


def cmd_work_plan(args):
    packet, taxonomies = load_context(args.packet)
    reviews = review_all(packet, taxonomies, as_of=args.as_of)
    plan = build_work_plan(packet, reviews, taxonomies)
    payload = plan.to_dict()
    if args.json:
        emit(payload, as_json=True)
        return
    print(f"== 锁定核查范围（{len(plan.locked_scope)} 件）：{plan.locked_scope} ==")
    for item in plan.items:
        print(
            f"- {item.sku} [{item.status}] {item.hours:g}h/{item.cost_class}"
            f"/误分代价 {item.error_cost}；覆盖买家 {item.affected_buyers}"
            f" 承诺 {item.affected_commitments}；{item.reason}"
        )
    print(f"\n容量 {plan.capacity_hours:g}h，高代价保底 {plan.reserve_high:g}h")
    print(f"本轮排入：{plan.scheduled}")
    print(f"延后（下轮优先）：{plan.deferred}")
    for row in plan.rollbacks:
        print(f"回退不重派：{row['sku']} 被 {row['superseded_by']['known']} 裁决覆盖")
    for row in plan.reconciliations:
        print(f"待对账：{row['sku']} 已于 {row['assigned']} 派出，新裁决 {row['new_ruling']['known']}")
    for note in plan.guardrail_notes:
        print(f"护栏：{note}")


def cmd_supplier(args):
    packet, taxonomies = load_context(args.packet)
    rows = supplier_view(packet, args.merchant, taxonomies)
    emit(rows, supplier_lines(rows, args.merchant), args.json)


def cmd_trace(args):
    packet, taxonomies = load_context(args.packet)
    reviews = review_all(packet, taxonomies, as_of=args.as_of)
    matched = [review for review in reviews if review.package == args.package]
    if not matched:
        raise SystemExit(f"没有组合 {args.package} 的在途承诺")
    emit(
        [review.to_dict() for review in matched],
        operator_commitment_lines(matched),
        args.json,
    )


def build_parser():
    parser = argparse.ArgumentParser(description="苹果商品可解释归类与组合承诺复查")
    parser.add_argument(
        "--packet", type=Path, default=DEFAULT_PACKET, help="资料包 JSON 路径"
    )
    parser.add_argument("--json", dest="json", action="store_true",
                        help="输出机器可读 JSON（子命令前后均可）")
    sub = parser.add_subparsers(dest="command", required=True)

    def add_json(p):
        # 默认值用 SUPPRESS，避免子解析器的 False 覆盖父解析器的 True
        p.add_argument("--json", dest="json", action="store_true",
                       default=argparse.SUPPRESS,
                       help="输出机器可读 JSON")
        return p

    p = sub.add_parser("classify", help="对 SKU 做归类（默认全部，人工裁决优先）")
    p.add_argument("--sku")
    p.add_argument("--taxonomy", choices=["v1", "v2"], default="v2")
    p.add_argument("--method", choices=["rule", "model"], default="model")
    p.add_argument("--as-of", help="只使用该日期（含）之前的裁决，格式 YYYY-MM-DD")
    p.set_defaults(func=cmd_classify)
    add_json(p)

    p = sub.add_parser("backtest", help="分组防泄漏回测与可靠度视图")
    p.set_defaults(func=cmd_backtest)
    add_json(p)

    p = sub.add_parser("review-orders", help="在途采购单复查，输出三类处置")
    p.add_argument("--as-of")
    p.set_defaults(func=cmd_review_orders)
    add_json(p)

    p = sub.add_parser("work-plan", help="人工复核范围锁定与排班")
    p.add_argument("--as-of")
    p.set_defaults(func=cmd_work_plan)
    add_json(p)

    p = sub.add_parser("supplier", help="供应商视图：适用标签与可供货版本")
    p.add_argument("--merchant", required=True)
    p.set_defaults(func=cmd_supplier)
    add_json(p)

    p = sub.add_parser("trace", help="追踪单个组合从可销售到待澄清的触发信息")
    p.add_argument("--package", required=True)
    p.add_argument("--as-of")
    p.set_defaults(func=cmd_trace)
    add_json(p)
    return parser


def main(argv=None):
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (OSError, ValueError, KeyError) as exc:
        parser.exit(2, f"无法处理资料：{exc}\n")


if __name__ == "__main__":
    main()
