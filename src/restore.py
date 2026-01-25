import re

KV_PATTERN = re.compile(r"(\S+?)=([^\s]+)")

def parse_kv_section(line: str) -> dict:
    return dict(KV_PATTERN.findall(line))

def extract_tokens_with_kv_filters(
    log_file: str,
    *,
    num_min=None,
    num_max=None,
    filters: dict[str, str | int] = None,
):
    results = []
    filters = filters or {}

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            kv = dict(KV_PATTERN.findall(line))
            if "num" not in kv or "tokens" not in kv:
                continue

            num = int(kv["num"])
            if num_min is not None and num < num_min:
                continue
            if num_max is not None and num > num_max:
                continue

            # 动态字段匹配
            ok = True
            for k, v in filters.items():
                if k not in kv:
                    ok = False
                    break
                if str(kv[k]) != str(v):
                    ok = False
                    break

            if not ok:
                continue

            results.append((num, int(kv["tokens"])))

    return results


import re
from typing import Optional

KV_PATTERN = re.compile(r"(\S+?)=([^\s]+)")

def exists_log_entry(
    log_file: str,
    *,
    num_min: Optional[int] = None,
    num_max: Optional[int] = None,
    dataset: Optional[str] = None,
    id_: Optional[int] = None,
    stage: Optional[str] = None,
    state: Optional[str] = None,
) -> bool:
    """
    只检查日志中是否存在满足条件的记录
    找到即返回 True，否则 False
    """

    with open(log_file, "r", encoding="utf-8") as f:
        for line in f:
            kv = dict(KV_PATTERN.findall(line))
            if not kv:
                continue

            # ===== num 区间判断（如果指定）=====
            if num_min is not None or num_max is not None:
                if "num" not in kv:
                    continue
                num = int(kv["num"])
                if num_min is not None and num < num_min:
                    continue
                if num_max is not None and num > num_max:
                    continue

            # ===== 其它字段判断 =====
            if dataset is not None and kv.get("dataset") != dataset:
                continue
            if id_ is not None and int(kv.get("id", -1)) != id_:
                continue
            if stage is not None and kv.get("stage") != stage:
                continue
            if state is not None and kv.get("state") != state:
                continue

            # ✔ 全部条件满足
            return True

    return False



with open("run.log") as f:
    for line in f:
        kv = parse_kv_section(line)
        if "num" in kv and "tokens" in kv:
            num = int(kv["num"])
            tokens = int(kv["tokens"])