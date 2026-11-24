"""资料读取与证据层。

设计约束（见 资料格式.md）：

- ``ingredients is None`` 表示没有已查配料，不等于“没有成分”，按证据缺失处理。
- 文案按 ``sku, revision`` 版本化；同一 SKU 改版属于同一商品，证据要带版次。
- 价格、转化等字段不作为标签真值，资料包中也没有这类字段。
- 证据分三类来源：配料事实（ingredient）、包装文案（copy）、人工裁决（ruling）。
  配料事实与文案相互独立，冲突时不做消解，只记录冲突，交给裁决/弃权逻辑。
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

# 配料事实 -> v2 叶类候选。配料是经过核查的事实，强于文案。
# 注意：切片/开袋/饮品/果浆 等词也可能出现在文案里，但两套映射按来源分开，
# 同一词在配料侧与文案侧命中同一叶类不算冲突，只有跨源指向不同叶类才算。
INGREDIENT_FACT_SIGNALS = {
    "纯苹果": "fresh",
    "富士": "fresh",
    "切片": "dry",
    "开袋": "dry",
    "饮品": "juice",
    "果浆": "juice",
}

# 文案 token -> v2 叶类候选。文案只是商家自述，弱证据。
COPY_TOKEN_SIGNALS = {
    "整果": "fresh",
    "鲜果": "fresh",
    "富士": "fresh",
    "纯苹果": "fresh",
    "干制": "dry",
    "果脯": "dry",
    "苹果干": "dry",
    "开袋": "dry",
    "切片": "dry",
    "果汁": "juice",
    "压榨": "juice",
    "饮品": "juice",
    "果浆": "juice",
}

# 出现这些 token 说明文案本身声明信息不全/未知。
UNKNOWN_COPY_TOKENS = frozenset({"未知"})

# 与品类无关的包装/营销词：名字里带“礼盒”不说明是纯果、混装还是果制品。
NON_EVIDENCE_TOKENS = frozenset(
    {"礼盒", "甄选", "果香", "手作", "风味", "工坊", "山园"}
)


def read_packet(path: str | Path) -> dict:
    """读取并做最外层结构校验，业务校验在 :class:`Packet` 构造时完成。"""
    with Path(path).open(encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict) or any(
        not isinstance(value, (list, dict)) for value in data.values()
    ):
        raise ValueError("资料包必须是由记录列表或参数对象组成的 JSON 对象")
    required = {
        "products",
        "text",
        "rulings",
        "taxonomies",
        "packages",
        "commitments",
        "review_work",
        "policy",
    }
    missing = required - data.keys()
    if missing:
        raise ValueError(f"资料包缺少资料集：{sorted(missing)}")
    return data


@dataclass(frozen=True)
class Evidence:
    """单条原子证据，带来源与版次，可直接展示给运营。"""

    source: str  # ingredient | copy(revision) | ruling(taxonomy)
    signal: Optional[str]  # v2 叶类候选；None 表示未知/信息不全
    detail: str
    weight: str  # fact | weak | human | absent

    def to_dict(self) -> dict:
        return {
            "source": self.source,
            "signal": self.signal,
            "detail": self.detail,
            "weight": self.weight,
        }


@dataclass
class Product:
    sku: str
    merchant: str
    family: str
    ingredients: Optional[list[str]]
    net_g: int
    launched: str
    texts: dict[int, dict] = field(default_factory=dict)
    rulings: list[dict] = field(default_factory=list)

    @property
    def latest_revision(self) -> int:
        return max(self.texts)

    def title_at(self, revision: int) -> str:
        return self.texts[revision]["title"]

    def body_at(self, revision: int) -> str:
        return self.texts[revision]["body"]

    def ingredient_evidence(self) -> list[Evidence]:
        if self.ingredients is None:
            return [
                Evidence(
                    source="ingredient",
                    signal=None,
                    detail="没有已查配料（缺失，不等于无成分）",
                    weight="absent",
                )
            ]
        out = []
        for item in self.ingredients:
            out.append(
                Evidence(
                    source="ingredient",
                    signal=INGREDIENT_FACT_SIGNALS.get(item),
                    detail=f"配料事实：{item}",
                    weight="fact",
                )
            )
        return out

    def copy_evidence(self, revision: Optional[int] = None) -> list[Evidence]:
        revision = revision or self.latest_revision
        row = self.texts[revision]
        tokens = row["title"].split() + row["body"].split()
        out = []
        unknown = any(token in UNKNOWN_COPY_TOKENS for token in tokens)
        for token in tokens:
            if token in NON_EVIDENCE_TOKENS or token.lstrip("山园").isdigit():
                continue
            signal = COPY_TOKEN_SIGNALS.get(token)
            if signal is None and token not in UNKNOWN_COPY_TOKENS:
                # 未登记 token 不猜，仅保留未知声明类证据
                continue
            out.append(
                Evidence(
                    source=f"copy@r{revision}",
                    signal=signal,
                    detail=(
                        f"文案声明信息不全：{token}"
                        if token in UNKNOWN_COPY_TOKENS
                        else f"文案词：{token}"
                    ),
                    weight="weak",
                )
            )
        if unknown and not any(e.signal for e in out):
            # 文案只有“未知/手作/礼盒”这类内容时，明确记录为信息不全
            out.append(
                Evidence(
                    source=f"copy@r{revision}",
                    signal=None,
                    detail=f"文案无可判定品类词：{row['title']} / {row['body']}",
                    weight="weak",
                )
            )
        return out

    def evidence_conflicts(self, revision: Optional[int] = None) -> list[frozenset[str]]:
        """配料事实与文案给出的互斥叶类候选；同源内部冲突也算。"""
        revision = revision or self.latest_revision
        by_source: dict[str, set[str]] = defaultdict(set)
        for evidence in self.ingredient_evidence():
            if evidence.signal:
                by_source[evidence.source].add(evidence.signal)
        for evidence in self.copy_evidence(revision):
            if evidence.signal:
                by_source[evidence.source].add(evidence.signal)
        signals = [signals for signals in by_source.values() if signals]
        conflicts = []
        for i, left in enumerate(signals):
            for right in signals[i + 1 :]:
                if left != right:
                    conflicts.append(frozenset(left | right))
        return conflicts

    def ruling_on(self, taxonomy: str, on_or_before: Optional[str] = None) -> Optional[dict]:
        """取某版次下截至某日的最后一条人工裁决；同版同料号后到记录修订前一条。"""
        rows = [
            row
            for row in self.rulings
            if row["taxonomy"] == taxonomy
            and (on_or_before is None or row["known"] <= on_or_before)
        ]
        if not rows:
            return None
        return max(rows, key=lambda row: row["known"])


class Packet:
    """整包资料的内存视图，负责引用完整性校验与索引。"""

    def __init__(self, data: dict):
        required = {
            "products",
            "text",
            "rulings",
            "taxonomies",
            "packages",
            "commitments",
            "review_work",
            "policy",
        }
        missing = required - data.keys()
        if missing:
            raise ValueError(f"资料包缺少资料集：{sorted(missing)}")
        self.data = data
        self.products: dict[str, Product] = {}
        for row in data["products"]:
            self.products[row["sku"]] = Product(
                sku=row["sku"],
                merchant=row["merchant"],
                family=row["family"],
                ingredients=row["ingredients"],
                net_g=row["net_g"],
                launched=row["launched"],
            )
        for row in data["text"]:
            self.products[row["sku"]].texts[row["revision"]] = row
        for row in data["rulings"]:
            self.products[row["sku"]].rulings.append(row)

        self._validate()
        self.groups = self._build_groups()
        self.skus_by_group: dict[str, list[str]] = defaultdict(list)
        for sku, group_id in self.groups.items():
            self.skus_by_group[group_id].append(sku)

    @classmethod
    def load(cls, path: str | Path) -> "Packet":
        return cls(read_packet(path))

    def _validate(self) -> None:
        skus = set(self.products)
        if len(skus) != len(self.data["products"]):
            raise ValueError("products 中 sku 重复")
        text_skus = {row["sku"] for row in self.data["text"]}
        if text_skus != skus:
            raise ValueError("text 与 products 的 sku 集合不一致")
        text_keys = [(row["sku"], row["revision"]) for row in self.data["text"]]
        if len(set(text_keys)) != len(text_keys):
            raise ValueError("text 中 (sku, revision) 不唯一")
        for row in self.data["rulings"] + self.data["review_work"]:
            if row["sku"] not in skus:
                raise ValueError(f"记录引用了不存在的 sku：{row['sku']}")
        for package in self.data["packages"]:
            members = package["base"] + sum(package["optional"], [])
            if len(set(members)) != len(members):
                raise ValueError(f"组合 {package['package']} 内部件重复")
            if any(sku not in skus for sku in members):
                raise ValueError(f"组合 {package['package']} 引用了不存在的 sku")
        package_ids = {row["package"] for row in self.data["packages"]}
        for row in self.data["commitments"]:
            if row["package"] not in package_ids:
                raise ValueError(f"承诺引用了不存在的组合：{row['package']}")

    def _build_groups(self) -> dict[str, str]:
        """防泄漏分组：同 family 或在同一 package 共同出现的 SKU 永不跨训练/考核边。"""
        parent = {sku: sku for sku in self.products}

        def find(x: str) -> str:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: str, b: str) -> None:
            root_a, root_b = find(a), find(b)
            if root_a != root_b:
                parent[root_b] = root_a

        families: dict[str, list[str]] = defaultdict(list)
        for product in self.products.values():
            families[product.family].append(product.sku)
        for members in families.values():
            for sku in members[1:]:
                union(members[0], sku)
        for package in self.data["packages"]:
            members = package["base"] + sum(package["optional"], [])
            for sku in members[1:]:
                union(members[0], sku)
        return {sku: find(sku) for sku in self.products}

    def package(self, package_id: str) -> dict:
        return next(
            row for row in self.data["packages"] if row["package"] == package_id
        )

    def label_counts(self, taxonomy: str) -> Counter:
        counts: Counter = Counter()
        for row in self.data["rulings"]:
            if row["taxonomy"] == taxonomy:
                counts[row["label"]] += 1
        return counts
