"""
Проверка на утечку доступов (пароли, ключи, строки подключения к БД) перед коммитом.

Запускается автоматически git-хуком .githooks/pre-commit для всех подготовленных к коммиту файлов.
Вручную:
    python tools/check_secrets.py --all            # весь репозиторий
    python tools/check_secrets.py путь/к/файлу ... # конкретные файлы или папки (например, перед добавлением)

Что проверяется:
  * текстовые файлы (.py, .ipynb, .sql, .md, .twb, .json, .yml, .ini ...);
  * содержимое Office/BI-архивов (.xlsx, .xlsm, .docx, .pptx, .twbx, .tdsx, .pbix, .pbit) —
    строки подключения Power Query, Tableau-соединения, внешние связи Excel;
  * имена файлов, которые не должны попадать в git вообще (.env, ключи, credentials.json ...).

Строку можно пометить как допустимую комментарием `secret-scan: ignore`.
Код выхода 1 = найдено, коммит блокируется.
"""
from __future__ import annotations

import re
import subprocess
import sys
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

FORBIDDEN_NAMES = re.compile(
    r"(^|/)(\.env(\.[\w-]+)?|.*\.pem|.*\.key|.*\.p12|.*\.pfx|id_rsa.*|\.pgpass|credentials\.json|"
    r"client_secret.*\.json|service[_-]?account.*\.json|secrets?\.(json|ya?ml|toml|ini|txt))$",
    re.I,
)
ALLOWED_NAMES = re.compile(r"(^|/)\.env\.example$", re.I)

RULES = [
    ("пароль в строке подключения", re.compile(r"(?i)\b(password|pwd|passwd)\s*=\s*(?!['\"]?\s*['\";]|\{|\$|<|\*{3})[^;'\"\s]{3,}")),
    ("пароль в коде/конфиге", re.compile(r"(?i)\b(password|passwd|pwd|db_pass(word)?)\b['\"]?\s*[:=]\s*['\"](?!\s*['\"])(?!\{|\$|<|your|xxx|\*{3}|changeme)[^'\"]{3,}['\"]")),
    ("логин:пароль в URL", re.compile(r"\b[a-z][a-z0-9+.-]*://[^/\s:@'\"]+:(?!\{|\$|<|\*)[^/\s@'\"]{2,}@[^\s'\"]+", re.I)),
    ("Tableau: сохранённый пароль", re.compile(r"(?i)\bpassword=['\"][^'\"]+['\"]")),
    ("API-ключ / токен", re.compile(r"(?i)\b(api[_-]?key|secret[_-]?key|access[_-]?token|auth[_-]?token|client[_-]?secret)\b['\"]?\s*[:=]\s*['\"][A-Za-z0-9_\-./+=]{12,}['\"]")),
    ("логин:пароль в адресе БД", re.compile(r"(?i)\b(mongodb(\+srv)?|mysql(\+\w+)?|postgres(ql)?(\+\w+)?|mssql(\+\w+)?|redis|amqp|jdbc:\w+)://[^/\s:@'\"]+:(?!\{|\$|<|\*)[^/\s@'\"]{3,}")),
    ("приватный ключ", re.compile(r"-----BEGIN (RSA |EC |OPENSSH |DSA |PGP )?PRIVATE KEY-----")),
    ("AWS access key", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}\b")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("Slack token", re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}\b")),
    ("OpenAI/Anthropic key", re.compile(r"\bsk-(ant-)?[A-Za-z0-9_\-]{20,}\b")),
]

TEXT_EXT = {".py", ".ipynb", ".sql", ".md", ".txt", ".json", ".yml", ".yaml", ".toml", ".ini", ".cfg",
            ".conf", ".env", ".sh", ".ps1", ".bat", ".r", ".twb", ".tds", ".xml", ".csv", ".html",
            ".js", ".m", ".pq", ".dax", ".config", ""}
ZIP_EXT = {".xlsx", ".xlsm", ".xltx", ".docx", ".pptx", ".twbx", ".tdsx", ".pbix", ".pbit"}
ZIP_MEMBER_EXT = {".xml", ".rels", ".twb", ".tds", ".json", ".m", ".txt", ".odc", ""}
MAX_TEXT = 20 * 1024 * 1024


def scan_text(text: str, where: str) -> list[str]:
    hits = []
    for n, line in enumerate(text.splitlines(), 1):
        if "secret-scan: ignore" in line or len(line) > 20000 and "base64" in line:
            continue
        for name, rx in RULES:
            m = rx.search(line)
            if m:
                frag = m.group(0)
                shown = frag[:6] + "…" if len(frag) > 6 else frag
                hits.append(f"{where}:{n}: {name} ({shown})")
    return hits


def scan_zip(path: Path, rel: str) -> list[str]:
    hits = []
    try:
        with zipfile.ZipFile(path) as z:
            for info in z.infolist():
                ext = Path(info.filename).suffix.lower()
                name = info.filename.lower()
                interesting = (ext in ZIP_MEMBER_EXT or "connections" in name or "datamashup" in name
                               or "externallink" in name or "connection" in name)
                if not interesting or info.file_size > MAX_TEXT:
                    continue
                raw = z.read(info)
                if "datamashup" in name:  # Power BI: внутри ещё один zip с M-кодом
                    start = raw.find(b"PK\x03\x04")
                    if start >= 0:
                        import io
                        try:
                            with zipfile.ZipFile(io.BytesIO(raw[start:])) as inner:
                                for m in inner.namelist():
                                    hits += scan_text(inner.read(m).decode("utf-8", "ignore"), f"{rel}!{info.filename}/{m}")
                        except zipfile.BadZipFile:
                            pass
                    continue
                for enc in ("utf-8", "utf-16"):
                    try:
                        hits += scan_text(raw.decode(enc), f"{rel}!{info.filename}")
                        break
                    except UnicodeDecodeError:
                        continue
    except zipfile.BadZipFile:
        pass
    return hits


def notebook_text(path: Path) -> str:
    """Код, markdown и выводы ячеек ноутбука как обычный текст (в .ipynb кавычки экранированы)."""
    import json
    try:
        nb = json.loads(path.read_text("utf-8", errors="ignore"))
    except ValueError:
        return path.read_text("utf-8", errors="ignore")
    parts = []
    for cell in nb.get("cells", []):
        parts.append("".join(cell.get("source", [])))
        for out in cell.get("outputs", []):
            parts.append("".join(out.get("text", [])))
            for v in out.get("data", {}).values():
                if isinstance(v, (list, str)) and not str(v).startswith("iVBOR"):
                    parts.append("".join(v) if isinstance(v, list) else v)
    return "\n".join(parts)


def scan_file(path: Path) -> list[str]:
    rel = path.relative_to(ROOT).as_posix() if path.is_relative_to(ROOT) else str(path)
    if FORBIDDEN_NAMES.search(rel) and not ALLOWED_NAMES.search(rel):
        return [f"{rel}: файл с секретами не должен попадать в git (добавьте в .gitignore)"]
    ext = path.suffix.lower()
    if ext in ZIP_EXT:
        return scan_zip(path, rel)
    if ext == ".ipynb":
        return scan_text(notebook_text(path), rel)
    if ext in TEXT_EXT and path.stat().st_size <= MAX_TEXT:
        return scan_text(path.read_text("utf-8", errors="ignore").replace('\\"', '"'), rel)
    return []


def staged_files() -> list[Path]:
    out = subprocess.run(["git", "diff", "--cached", "--name-only", "--diff-filter=ACMR", "-z"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [ROOT / p for p in out.split("\0") if p]


def tracked_and_new() -> list[Path]:
    out = subprocess.run(["git", "ls-files", "-co", "--exclude-standard", "-z"],
                         cwd=ROOT, capture_output=True, text=True, check=True).stdout
    return [ROOT / p for p in out.split("\0") if p]


def expand(args: list[str]) -> list[Path]:
    files = []
    for a in args:
        p = Path(a).resolve()
        files += [f for f in p.rglob("*") if f.is_file()] if p.is_dir() else [p]
    return files


def main(argv: list[str]) -> int:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # консоль Windows (cp1252)
    if not argv or argv == ["--staged"]:
        files = staged_files()
    elif argv == ["--all"]:
        files = tracked_and_new()
    else:
        files = expand(argv)
    hits = []
    for f in files:
        if f.exists() and f.name != "check_secrets.py":
            hits += scan_file(f)
    if hits:
        print("⛔ Найдены возможные доступы/секреты — коммит остановлен:\n")
        print("\n".join(f"  {h}" for h in hits))
        print("\nУберите их (перенесите в .env, см. .env.example) или, если это ложное срабатывание,"
              "\nдобавьте в строку комментарий `secret-scan: ignore`.")
        return 1
    print(f"✔ Проверено файлов: {len(files)}, доступов не найдено.")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
