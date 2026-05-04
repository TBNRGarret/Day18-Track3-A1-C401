"""
Kiểm tra định dạng bài nộp trước khi submit.
Chạy: python check_lab.py

⚠️ Lỗi định dạng khiến script chấm tự động không chạy → trừ 5 điểm thủ tục.
"""

import json
import os
import sys
import subprocess


def check_file(path: str, required: bool = True) -> bool:
    if os.path.exists(path):
        print(f"  ✅ {path}")
        return True
    elif required:
        print(f"  ❌ THIẾU: {path}")
        return False
    else:
        print(f"  ⚠️  Optional: {path}")
        return True


def check_json(path: str, required_keys: list[str]) -> bool:
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        missing = [k for k in required_keys if k not in data]
        if missing:
            print(f"  ❌ {path} thiếu keys: {missing}")
            return False
        print(f"  ✅ {path} — keys OK")
        return True
    except (json.JSONDecodeError, FileNotFoundError) as e:
        print(f"  ❌ {path} — {e}")
        return False


def check_todos() -> int:
    """Count remaining TODO markers in src/."""
    count = 0
    for root, _, files in os.walk("src"):
        for f in files:
            if f.endswith(".py"):
                with open(os.path.join(root, f), encoding="utf-8") as fh:
                    for line in fh:
                        if "# TODO:" in line:
                            count += 1
    return count


def run_tests() -> tuple[int, int]:
    """Run pytest and return (passed, total). total = passed + failed (bỏ qua skipped)."""
    import re

    try:
        # Ghi đè addopts trong pytest.ini (vd. -v) để dòng tóm tắt không bị bọc trong "====".
        result = subprocess.run(
            [
                sys.executable,
                "-m",
                "pytest",
                "tests/",
                "--tb=no",
                "-q",
                "-o",
                "addopts=",
            ],
            capture_output=True,
            text=True,
            timeout=600,
        )
    except Exception as e:
        print(f"  ⚠️  pytest error: {e}")
        return 0, 0

    out = (result.stdout or "") + "\n" + (result.stderr or "")

    def parse_summary(text: str) -> tuple[int, int]:
        passed = failed = 0
        for line in reversed(text.strip().split("\n")):
            line_stripped = line.strip()
            if not line_stripped or set(line_stripped) <= {"=", "-", "_"}:
                continue
            if "passed" not in line_stripped and "failed" not in line_stripped:
                continue
            # Bỏ qua header/tiêu đề phụ
            if "test session starts" in line_stripped.lower():
                continue
            if "warnings summary" in line_stripped.lower():
                continue

            pm = re.search(r"(\d+)\s+passed", line_stripped)
            fm = re.search(r"(\d+)\s+failed", line_stripped)
            if pm:
                passed = int(pm.group(1))
            if fm:
                failed = int(fm.group(1))
            if pm or fm:
                return passed, passed + failed
        return 0, 0

    passed, total = parse_summary(out)
    if total == 0 and result.returncode != 0 and "error" in out.lower():
        # pytest collection error / crash — không có dòng "N passed"
        err_line = next(
            (ln for ln in out.split("\n") if "ERROR" in ln or "ImportError" in ln),
            "xem stdout pytest",
        )
        print(f"  ⚠️  pytest error: {err_line.strip()[:120]}")
    elif total == 0 and passed == 0:
        err_line = next(
            (ln for ln in reversed(out.split("\n")) if ln.strip()),
            "no output",
        )
        if "no tests ran" in out.lower() or "collected 0" in out.lower():
            print("  ⚠️  pytest error: không thu thập được test nào")
        elif err_line and err_line != "no output":
            print(f"  ⚠️  pytest error: không parse được summary — {err_line.strip()[:100]}")

    return passed, total


def validate():
    print("🔍 Kiểm tra bài nộp Lab 18: Production RAG\n")
    errors = 0

    # 1. Source files
    print("📁 Source code:")
    for f in ["src/m1_chunking.py", "src/m2_search.py", "src/m3_rerank.py",
              "src/m4_eval.py", "src/pipeline.py"]:
        if not check_file(f):
            errors += 1

    # 2. Reports
    print("\n📊 Reports:")
    if check_file("reports/ragas_report.json"):
        if not check_json("reports/ragas_report.json", ["aggregate", "num_questions"]):
            errors += 1
    else:
        errors += 1
    check_file("reports/naive_baseline_report.json", required=False)

    # 3. Analysis
    print("\n📝 Analysis:")
    check_file("analysis/failure_analysis.md")
    check_file("analysis/group_report.md")

    # 4. Individual reflections
    print("\n👤 Individual reflections:")
    reflections = []
    ref_dir = "analysis/reflections"
    if os.path.isdir(ref_dir):
        reflections = [f for f in os.listdir(ref_dir) if f.startswith("reflection_") and f.endswith(".md")]
    if reflections:
        for r in reflections:
            print(f"  ✅ {ref_dir}/{r}")
    else:
        print(f"  ⚠️  Chưa có file reflection cá nhân trong {ref_dir}/")

    # 5. TODO count
    print("\n🔧 TODO markers:")
    todo_count = check_todos()
    if todo_count == 0:
        print("  ✅ Không còn TODO nào")
    else:
        print(f"  ⚠️  Còn {todo_count} TODO chưa implement")

    # 6. Tests
    print("\n🧪 Auto-tests:")
    passed, total = run_tests()
    if total > 0:
        pct = passed / total * 100
        print(f"  {'✅' if pct >= 80 else '⚠️'} {passed}/{total} tests passed ({pct:.0f}%)")
    else:
        print("  ⚠️  Không chạy được tests")

    # 7. Summary
    print("\n" + "=" * 50)
    if errors == 0:
        print("🚀 Bài lab sẵn sàng để nộp!")
    else:
        print(f"❌ Có {errors} lỗi. Sửa trước khi nộp.")
    print("=" * 50)


if __name__ == "__main__":
    validate()
