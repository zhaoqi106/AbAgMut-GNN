import os
import shutil
import subprocess
import tempfile
from multiprocessing import Pool

import pandas as pd
from Bio.PDB import PDBParser, PDBIO, Select


# ========= 路径写死（按你给的） =========
INPUT_DIR = "/home/zhao/gwc/NEW3/data/S487/wt"
CSV_PATH  = "/home/zhao/gwc/NEW3/data/S487/S487.csv"
OUTPUT_DIR = "/home/zhao/gwc/NEW3/data/S487/repairpdb"

CSV_COL = "PDB"

# 你机器上的 FoldX 路径：需要你确认这一行是否正确
FOLDX_EXE = r"/home/zhao/gwc/MODEL/FOLDX/foldx"


# ========= 工具函数 =========
def normalize_id(x: str) -> str:
    """统一ID：去空白、小写、去掉 .pdb 后缀 -> 返回不带后缀的 pdb id"""
    if x is None:
        return ""
    s = str(x).strip().lower()
    if s.endswith(".pdb"):
        s = s[:-4]
    return s


def load_pdb_ids_from_csv(csv_path: str, col: str) -> list[str]:
    df = pd.read_csv(csv_path)
    if col not in df.columns:
        raise ValueError(f"CSV 中找不到列: {col}，现有列: {list(df.columns)}")

    ids = []
    for v in df[col].dropna().tolist():
        pid = normalize_id(v)
        if pid:
            ids.append(pid)
    # 去重但保持顺序
    seen = set()
    out = []
    for x in ids:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return out


# ========= 大小写不敏感的文件查找 =========
def find_file_case_insensitive(directory: str, filename: str) -> str:
    """
    在目录中查找文件（大小写不敏感）
    返回实际的文件路径，如果找不到返回None
    """
    if not os.path.isdir(directory):
        return None
    
    # 先尝试直接匹配（最快）
    direct_path = os.path.join(directory, filename)
    if os.path.isfile(direct_path):
        return direct_path
    
    # 大小写不敏感搜索
    filename_lower = filename.lower()
    for item in os.listdir(directory):
        if item.lower() == filename_lower:
            full_path = os.path.join(directory, item)
            if os.path.isfile(full_path):
                return full_path
    
    return None


# ========= Biopython 清洗：去 HETATM（只在临时目录生成） =========
class NonHetSelect(Select):
    def accept_residue(self, residue):
        return residue.id[0] == ' '


def clean_to_file(input_pdb: str, output_pdb: str):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("PDB", input_pdb)
    io = PDBIO()
    io.set_structure(structure)
    io.save(output_pdb, NonHetSelect())


def run_foldx_repair(foldx_exe: str, workdir: str, pdb_filename: str):
    cmd = [foldx_exe, "--command=RepairPDB", f"--pdb={pdb_filename}"]
    # 不捕获输出，让 FoldX 直接打印到控制台，避免挂起
    # 添加 300 秒超时（5分钟），可根据需要调整
    try:
        subprocess.run(cmd, cwd=workdir, check=True, timeout=6000)
    except subprocess.TimeoutExpired:
        raise RuntimeError(f"FoldX 执行超时（>6000秒）: {pdb_filename}")


def find_repair_output(workdir: str, cleaned_basename: str) -> str:
    """
    FoldX 常见输出：<name>_Repair.pdb
    这里严格优先找该文件；找不到再兜底找唯一的 *_Repair.pdb
    """
    root = cleaned_basename[:-4] if cleaned_basename.lower().endswith(".pdb") else cleaned_basename
    expected = os.path.join(workdir, f"{root}_Repair.pdb")
    if os.path.isfile(expected):
        return expected

    candidates = [f for f in os.listdir(workdir) if f.lower().endswith("_repair.pdb")]
    if len(candidates) == 1:
        return os.path.join(workdir, candidates[0])

    raise FileNotFoundError(
        f"找不到 Repair 输出。期望: {os.path.basename(expected)}；实际候选: {candidates}"
    )


def process_single_pdb(args):
    """
    处理单个PDB文件的函数（用于多进程）
    返回: (pid, status, message)
    status: 'OK', 'SKIP', 'MISSING', 'FAIL'
    """
    pid, input_dir, output_dir, foldx_exe, total, index = args
    
    # 使用大小写不敏感的文件查找
    in_pdb = find_file_case_insensitive(input_dir, f"{pid}.pdb")
    # 输出文件名改为大写，后缀保持小写 .pdb
    out_pdb = os.path.join(output_dir, f"{pid.upper()}.pdb")
    
    # 检查输入文件是否存在
    if in_pdb is None:
        msg = f"[{index}/{total}] MISSING: {os.path.join(input_dir, pid)}.pdb (searched case-insensitive)"
        print(msg)
        return (pid, 'MISSING', msg)
    
    # 如果已经生成过就跳过
    if os.path.isfile(out_pdb):
        msg = f"[{index}/{total}] SKIP (exists): {out_pdb}"
        print(msg)
        return (pid, 'SKIP', msg)
    
    try:
        with tempfile.TemporaryDirectory(prefix="atlas_foldx_") as tmpdir:
            cleaned_basename = f"{pid}.pdb"
            cleaned_path = os.path.join(tmpdir, cleaned_basename)
            
            # 1) 清洗到临时目录（使用找到的实际文件路径）
            clean_to_file(in_pdb, cleaned_path)
            
            # 2) FoldX repair（在临时目录里跑）
            run_foldx_repair(foldx_exe, tmpdir, cleaned_basename)
            
            # 3) 找到 *_Repair.pdb，把它 move 成"原名.pdb"到输出目录
            repaired_tmp = find_repair_output(tmpdir, cleaned_basename)
            shutil.move(repaired_tmp, out_pdb)
        
        msg = f"[{index}/{total}] OK: {pid} -> {out_pdb}"
        print(msg)
        return (pid, 'OK', msg)
    
    except subprocess.CalledProcessError as e:
        stderr_msg = e.stderr.strip() if e.stderr else "No stderr"
        msg = f"[{index}/{total}] FAIL(FoldX): {pid} | stderr: {stderr_msg}"
        print(msg)
        return (pid, 'FAIL', msg)
    
    except Exception as e:
        msg = f"[{index}/{total}] FAIL: {pid} | {e}"
        print(msg)
        return (pid, 'FAIL', msg)


def main():
    # 基本检查
    if not os.path.isfile(CSV_PATH):
        raise FileNotFoundError(f"CSV 不存在: {CSV_PATH}")
    if not os.path.isdir(INPUT_DIR):
        raise FileNotFoundError(f"输入目录不存在: {INPUT_DIR}")
    if not os.path.isfile(FOLDX_EXE):
        raise FileNotFoundError(f"FoldX 不存在: {FOLDX_EXE}")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pdb_ids = load_pdb_ids_from_csv(CSV_PATH, CSV_COL)
    total = len(pdb_ids)
    
    print(f"\n开始处理 {total} 个PDB文件，使用 10 进程并行...")
    print(f"INPUT_DIR: {INPUT_DIR}")
    print(f"OUTPUT_DIR: {OUTPUT_DIR}")
    print(f"FOLDX_EXE: {FOLDX_EXE}\n")

    # 准备参数列表
    args_list = [(pid, INPUT_DIR, OUTPUT_DIR, FOLDX_EXE, total, i+1) 
                 for i, pid in enumerate(pdb_ids)]

    # 使用多进程池处理
    with Pool(processes=10) as pool:
        results = pool.map(process_single_pdb, args_list)

    # 统计结果
    ok = sum(1 for _, status, _ in results if status == 'OK')
    skip = sum(1 for _, status, _ in results if status == 'SKIP')
    missing = sum(1 for _, status, _ in results if status == 'MISSING')
    failed = sum(1 for _, status, _ in results if status == 'FAIL')

    # 写入日志
    log_path = os.path.join(OUTPUT_DIR, "repair_log.txt")
    with open(log_path, "w", encoding="utf-8") as log:
        log.write(f"INPUT_DIR={INPUT_DIR}\n")
        log.write(f"CSV_PATH={CSV_PATH}\n")
        log.write(f"OUTPUT_DIR={OUTPUT_DIR}\n")
        log.write(f"FOLDX_EXE={FOLDX_EXE}\n")
        log.write(f"PROCESSES=10\n\n")
        
        for pid, status, msg in results:
            log.write(msg + "\n")
        
        log.write("\n")
        log.write(f"TOTAL={total}, OK={ok}, SKIP={skip}, MISSING={missing}, FAILED={failed}\n")

    print("\n==== Summary ====")
    print("TOTAL  :", total)
    print("OK     :", ok)
    print("SKIP   :", skip)
    print("MISSING:", missing)
    print("FAILED :", failed)
    print("LOG    :", log_path)


if __name__ == "__main__":
    main()
