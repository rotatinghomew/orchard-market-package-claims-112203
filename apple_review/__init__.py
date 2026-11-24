"""苹果商品归类、组合承诺复查与人工复核的业务应用。

仅使用 Python 标准库。模块划分：

- evidence   资料读取、版本化事实、证据抽取、防泄漏分组
- taxonomy   类目版次、跨版兼容、父类标签不得当作新版子类真值的规则
- classifier 可解释规则参照做法、带弃权的多分类朴素贝叶斯做法
- evaluation 按历史截止点的分组切分与可靠度视图
- commitment 组件不确定性向组合级承诺的传播与处置分级
- workload   人工复核范围锁定、共享事实排序、回退不重派
- reporting  运营/供应商两种业务视图
"""

__all__ = [
    "evidence",
    "taxonomy",
    "classifier",
    "evaluation",
    "commitment",
    "workload",
    "reporting",
]
