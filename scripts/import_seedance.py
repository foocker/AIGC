"""Import the pinned multimodal-journey reading material into editable TeX.

The delivered TeX compiles without this script, Pandoc, or the source repository.
Re-import is explicit: it replaces only generated files, never primer.tex.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path
from urllib.parse import quote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
TITLES = [
    "公开能力与证据边界", "因果视频 VAE 与时空压缩", "流匹配与采样方向",
    "时空 DiT 与多模态注意力", "多模态位置编码与媒体时钟", "音视频联合生成",
    "长上下文与缓存条件", "多模态参考控制", "视频局部编辑",
    "后训练、偏好与奖励", "推理优化与瓶颈分析", "训练数据的制作与隔离",
    "Wan 基座的增量候选架构", "增量训练阶段与停止条件", "业务映射与交付成本",
    "机制代码与真实工程路线", "多参考绑定与时间控制", "源条件与统一编辑接口",
    "白模、相机与场景记忆", "完整联合音视频候选架构", "联合模型训练课程",
    "新增模块的维度与接入契约", "主体槽、参考冲突与细节预算", "音视频编辑状态与四类掩码",
    "长视频执行与因果缓存协议", "联合数据生产、噪声课程与参数组", "推理资源与交付预算",
    "Seedance 2.5 完整推测方案与投入选择",
]
PARTS = {
    0: "数学基础与生成模块",
    11: "数据、训练与增量路线",
    16: "条件控制与联合系统",
    21: "实现规格与完整推测方案",
}


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def tex(s: str) -> str:
    chars = {"\\": r"\textbackslash{}", "&": r"\&", "%": r"\%", "$": r"\$",
             "#": r"\#", "_": r"\_", "{": r"\{", "}": r"\}",
             "~": r"\textasciitilde{}", "^": r"\textasciicircum{}"}
    return "".join(chars.get(c, c) for c in str(s))


def plain(obj) -> str:
    if isinstance(obj, list):
        return " ".join(plain(x) for x in obj)
    if not isinstance(obj, dict):
        return ""
    if obj.get("t") in ("Str",):
        return obj["c"]
    if obj.get("t") in ("Code", "Math"):
        return obj["c"][-1]
    return plain(obj.get("c", []))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=ROOT.parent / "multimodal-journey")
    args = parser.parse_args()
    source = args.source.resolve()
    try:
        pandoc = shutil.which("pandoc")
        if not pandoc:
            import pypandoc
            pandoc = pypandoc.get_pandoc_path()
    except ImportError as exc:
        raise SystemExit("Install pypandoc_binary for optional re-import.") from exc
    commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    dirty = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True)
    base_url = f"https://github.com/foocker/multimodal-journey/blob/{commit}/"
    catalog = json.loads((source / "articles.json").read_text(encoding="utf-8"))["articles"]
    sources = {}
    for file, key in [("papers.json", "papers"), ("web.json", "sources"), ("code.json", "repositories")]:
        for item in json.loads((source / "sources" / file).read_text(encoding="utf-8"))[key]:
            sources[item.get("id", item.get("source_id"))] = item
    figures = json.loads((source / "assets/paper-figures/manifest.json").read_text(encoding="utf-8"))["figures"]
    figure_map = {(source / x["path"]).resolve(): x for x in figures}
    chapter_map = {(source / x["path"]).resolve(): "seedance-" + x["id"].lower() for x in catalog}
    examples = sorted((source / "algorithms/examples").glob("*.py"))
    code_map = {p.resolve(): "code-" + p.stem.replace("_", "-") for p in examples}
    out = ROOT / "chapters/seedance"
    image_dir = ROOT / "assets/seedance/figures"
    provenance = ROOT / "research/seedance-import"
    for path in (out, image_dir, provenance):
        path.mkdir(parents=True, exist_ok=True)
    imported, used_ids, used_figures = [], set(), {}
    counts = {"math": 0, "tables": 0, "code_blocks": 0, "images": 0}

    def run_pandoc(args, content):
        proc = subprocess.run([str(pandoc), *args], input=content, encoding="utf-8", text=True,
                              capture_output=True, check=True)
        if proc.stderr.strip():
            print(proc.stderr.strip())
        return proc.stdout

    for index, article in enumerate(catalog):
        path = (source / article["path"]).resolve()
        body = path.read_text(encoding="utf-8")
        metadata = re.match(r"^---\s*\n(.*?)\n---\s*\n", body, re.S)
        ids_match = re.search(r"^source_ids:\s*(\[.*\])", metadata[1] if metadata else "", re.M)
        ids = json.loads(ids_match[1]) if ids_match else []
        missing = set(ids) - sources.keys()
        if missing:
            raise ValueError(f"Unknown sources in {path}: {missing}")
        used_ids.update(ids)
        doc = json.loads(run_pandoc(["-f", "markdown+tex_math_dollars", "-t", "json"], body))
        doc["meta"] = {}
        prefix = chapter_map[path]

        def walk(node):
            if isinstance(node, list):
                return [walk(x) for x in node]
            if not isinstance(node, dict):
                return node
            node = copy.deepcopy(node)
            kind, c = node.get("t"), node.get("c")
            if kind == "Header":
                if c[0] == 1:
                    c[1][0] = prefix
                    c[2] = [{"t": "Str", "c": f"{article['id']} {TITLES[index]}"}]
                else:
                    c[1][0] = prefix + "-" + c[1][0]
            elif kind == "Math":
                counts["math"] += 1
            elif kind == "Code":
                # xurl permits breaks in long file paths and tensor layouts.
                literal = c[1].replace("{", "\\{").replace("}", "\\}")
                if not any(x in literal for x in ("\\", "{", "}", "%", "#")):
                    return {"t": "RawInline", "c": ["latex", r"\codepath{" + literal + "}"]}
            elif kind == "CodeBlock":
                counts["code_blocks"] += 1
                # Preserve the content but normalize diagonal diagram strokes to glyphs in Consolas.
                c[1] = c[1].replace("╲", "\\").replace("╱", "/")
            elif kind == "Link":
                target = c[2][0]
                split = urlsplit(target)
                if not split.scheme and not target.startswith("//"):
                    if target.startswith("#"):
                        c[2][0] = "#" + prefix + "-" + target[1:]
                    else:
                        resolved = (path.parent / split.path).resolve()
                        label = chapter_map.get(resolved) or code_map.get(resolved)
                        if label:
                            c[2][0] = "#" + label
                        else:
                            rel = resolved.relative_to(source).as_posix()
                            c[2][0] = base_url + quote(rel) + (("#" + split.fragment) if split.fragment else "")
            elif kind == "Image":
                counts["images"] += 1
                original = (path.parent / c[2][0]).resolve()
                entry = figure_map[original]
                destination = image_dir / original.name
                shutil.copy2(original, destination)
                used_figures[entry["id"]] = entry
                used_ids.add(entry["source_id"])
                c[2][0] = destination.relative_to(ROOT).as_posix()
                c[1].extend([{"t": "Space"}, {"t": "Str", "c": f"（论文示意，来源：{entry['source_id']}）"}])
            elif kind == "Figure":
                # Pandoc stores the visible caption separately from image alt text.
                image_nodes = []
                def collect_images(value):
                    if isinstance(value, dict):
                        if value.get("t") == "Image":
                            image_nodes.append(value)
                        collect_images(value.get("c", []))
                    elif isinstance(value, list):
                        for child in value:
                            collect_images(child)
                collect_images(c[2])
                if image_nodes:
                    original = (path.parent / image_nodes[0]["c"][2][0]).resolve()
                    sid = figure_map[original]["source_id"]
                    caption_blocks = c[1][1]
                    caption_blocks[-1]["c"].extend([
                        {"t": "Space"}, {"t": "Str", "c": f"（论文示意，来源：{sid}）"}])
            elif kind == "Table":
                counts["tables"] += 1
                # Explicit column widths make Chinese cells wrap instead of overflowing l columns.
                n = len(c[2])
                weights = ([0.22, 0.78] if n == 2 else [0.19, 0.39, 0.42] if n == 3
                           else [0.16] + [0.84 / (n - 1)] * (n - 1))
                for col, weight in zip(c[2], weights):
                    col[1] = {"t": "ColWidth", "c": weight}
            if "c" in node:
                node["c"] = walk(node["c"])
            return node

        doc["blocks"] = walk(doc["blocks"])
        latex = run_pandoc(["-f", "json", "-t", "latex", "--top-level-division=chapter",
                            "--syntax-highlighting=none", "--wrap=none"], json.dumps(doc, ensure_ascii=False))
        if index == 0:
            category = "公开能力梳理；沿用来源仓库既有记录"
        elif index <= 11:
            category = "基础方法与机制解释"
        elif index <= 15:
            category = "增量路线与工程阅读；未运行教学代码"
        else:
            category = "候选设计与工程选型；组合效果待验证"
        source_links = "、".join(r"\hyperref[src-" + sid + r"]{\texttt{" + tex(sid) + "}}" for sid in ids)
        marker = re.search(r"\\label\{[^}]+\}", latex)
        if marker is None:
            raise ValueError("Missing chapter label")
        latex = latex[:marker.end()] + "\n\\moduleinfo{" + article["id"] + "}{" + category + "}{" + source_links + "}\n" + latex[marker.end():]
        latex = latex.replace(r"\begin{longtable}", r"\small\setlength{\tabcolsep}{3pt}\renewcommand{\arraystretch}{1.18}" + "\n" + r"\begin{longtable}")
        latex = latex.replace(r"\def\LTcaptype{none}", r"\def\LTcaptype{table}")
        # TeX supplies the section number; omit numbering embedded in Markdown titles.
        latex = re.sub(r"(\\(?:sub)*section\{)\d+\.\s*", r"\1", latex)
        # Generated from Markdown; edit the .tex directly for local review, or explicitly re-import.
        destination = out / (article["id"].lower() + ".tex")
        destination.write_text("% Imported from " + article["path"] + "\n" + latex, encoding="utf-8")
        imported.append({**article, "source_sha256": sha(path), "tex_path": destination.relative_to(ROOT).as_posix(),
                         "source_ids": ids})

    code = [r"\chapter{极简机制代码}", r"\label{seedance-code}",
            "本章收录来源仓库自行编写的16份教学机制片段，以便离线对照正文。代码省略生产接口与边界处理，未在本次任务中执行；它们不是完整训练框架或闭源模型实现。"]
    for path in examples:
        contents = path.read_text(encoding="utf-8")
        code.append(
            r"\Needspace{12\baselineskip}" + "\n" + r"\section{" + tex(path.name) + "}\n" +
            r"\label{" + code_map[path.resolve()] + "}\n" + r"\begin{verbatim}" + "\n" +
            contents.rstrip() + "\n" + r"\end{verbatim}")
    (out / "mechanisms.tex").write_text("\n\n".join(code) + "\n", encoding="utf-8")

    refs = [r"\chapter{来源、固定版本与插图出处}", r"\label{seedance-sources}",
            "本章从来源仓库导入登记信息，用于追溯正文的基础方法和候选设计。论文结论受原实验设置限制，引用不代表本文组合方案已经验证。网页状态沿用2026年11月22日记录；本次未刷新线上产品与价格。各模块仍为审阅稿。",
            r"\section{论文与网页}"]
    for sid in sorted(used_ids):
        entry = sources[sid]
        if "repository" in entry:
            continue
        version = entry.get("edition", entry.get("type", ""))
        if entry.get("arxiv_id"):
            version += "; arXiv " + entry["arxiv_id"] + (entry.get("arxiv_version") or "")
        version += "; 来源登记检查日 " + entry.get("last_checked", entry.get("checked_on", "未登记"))
        refs.append(r"\sourceentry{" + sid + "}{" + tex(entry.get("title", sid)) + "}{" + tex(version) + "}{" + entry.get("url", entry.get("arxiv_url", "")) + "}")
        if entry.get("notes"):
            refs.append("{\\small 支持范围：" + tex(entry["notes"]) + "\\par}")
    refs.append(r"\section{上游代码与模型身份}")
    for sid in sorted(used_ids):
        entry = sources[sid]
        if "repository" not in entry:
            continue
        detail = "固定 commit " + entry["commit"] + "；代码许可 " + entry.get("code_license", "按固定版本核查")
        refs.append(r"\sourceentry{" + sid + "}{" + tex(entry["repository"]) + "}{" + tex(detail) + "}{" + entry["url"] + "/tree/" + entry["commit"] + "}")
        refs.append("{\\small 支持范围：" + tex(entry.get("limits", entry.get("purpose", ""))) + "\\par}")
    refs.extend([r"\section{论文插图}", "插图是原论文的局部示意，不能当作本书候选系统或 Seedance 2.5 官方架构。以下页序从1开始；裁剪坐标以 PDF point 为单位。原 PDF 哈希与完整坐标随附于插图 manifest，未复制整篇论文。",
                 r"\begin{longtable}{@{}p{0.35\linewidth}p{0.25\linewidth}p{0.30\linewidth}@{}}",
                 r"\toprule 插图 & 来源 ID & PDF页序与裁剪范围 \\ \midrule\endhead"])
    for fid, entry in sorted(used_figures.items()):
        refs.append(tex(fid) + " & " + tex(entry["source_id"]) + " & " + tex(str(entry["pdf_page"]) + "; " + str(entry["crop_points"])) + r" \\")
    refs.extend([r"\bottomrule\end{longtable}", r"\section{本次导入身份}",
                 "来源仓库：\\url{https://github.com/foocker/multimodal-journey}。",
                 "固定 commit：\\codepath{" + commit + "}。导入日：2026年11月24日。",
                 "逐模块源文件 SHA-256、图件来源、原模型元数据与新旧文件对应关系，见随附的 "
                 r"\codepath{research/seedance-import/manifest.json}。这份记录是文件追溯，不能代替科学复现。"])
    (out / "sources.tex").write_text("\n\n".join(refs) + "\n", encoding="utf-8")
    contents = ["% Shared contents: standalone and integrated book.", r"\input{chapters/seedance/primer}"]
    for index, article in enumerate(catalog):
        if index in PARTS:
            contents.append(r"\part{" + PARTS[index] + "}")
        contents.append(r"\input{chapters/seedance/" + article["id"].lower() + "}")
    contents.extend([r"\part{机制代码与来源}", r"\input{chapters/seedance/mechanisms}",
                     r"\input{chapters/seedance/sources}"])
    (out / "contents.tex").write_text("\n".join(contents) + "\n", encoding="utf-8")
    copies = ["sources/papers.json", "sources/web.json", "sources/code.json", "assets/paper-figures/manifest.json",
              "topics/video-generation/seedance-2.5/design-spec.json",
              "research/2026-11-22-implementation-contracts/model-config-check.json"]
    for relative in copies:
        destination = provenance / ("figure-manifest.json" if "paper-figures" in relative else Path(relative).name)
        shutil.copy2(source / relative, destination)
    manifest = {"imported_on": "2026-11-24", "status": "draft", "source_repository": "https://github.com/foocker/multimodal-journey",
                "source_commit": commit, "source_worktree_dirty": bool(dirty.strip()),
                "aigc_base_commit": "0a5ee3d50e9fb6a6d1d3f0a44c1dc8364c9485d0",
                "scope": "Editorial TeX import; existing evidence records reused, no fresh model experiment or online product verification.",
                "counts": counts, "modules": imported,
                "mechanisms": [{"path": p.relative_to(source).as_posix(), "sha256": sha(p)} for p in examples],
                "figures": list(used_figures.values())}
    (provenance / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"modules": len(imported), "sources": len(used_ids), "unique_figures": len(used_figures), **counts}))


if __name__ == "__main__":
    main()
