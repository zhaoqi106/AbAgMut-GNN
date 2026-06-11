import os
import copy
import signal
import ctypes
import tempfile
import subprocess
from pathlib import Path
import multiprocessing as mp

import pandas as pd
from Bio.PDB import PDBParser, is_aa
from Bio.PDB.SASA import ShrakeRupley
from Bio.PDB.DSSP import make_dssp_dict, residue_max_acc

from config import DATA_CONFIG, DSSP_CONFIG
import warnings

warnings.filterwarnings(
    "ignore",
    category=FutureWarning,
    message=r".*torch\.load.*weights_only=False.*"
)
_POOL = None


def _set_pdeathsig(sig=signal.SIGTERM):
    try:
        libc = ctypes.CDLL("libc.so.6")
        PR_SET_PDEATHSIG = 1
        libc.prctl(PR_SET_PDEATHSIG, int(sig))
    except Exception:
        pass


def init_worker():
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    _set_pdeathsig(signal.SIGTERM)
    try:
        signal.signal(signal.SIGHUP, signal.SIG_DFL)
    except Exception:
        pass


def normalize_icode(icode) -> str:
    if icode is None:
        return " "
    s = str(icode).strip()
    return s[0] if s else " "


def parse_resid_to_resseq_icode(resid):
    icode = " "
    resseq = None

    if isinstance(resid, tuple):
        for x in resid:
            if isinstance(x, int):
                resseq = x
                break
        for x in resid:
            if isinstance(x, str) and len(x) == 1 and x.isalpha():
                icode = x
                break
        if resseq is None:
            for x in resid:
                if isinstance(x, str) and x.lstrip("-").isdigit():
                    resseq = int(x)
                    break
        if resseq is None:
            return None, None
        return resseq, normalize_icode(icode)

    if isinstance(resid, int):
        return resid, " "

    s = str(resid).strip()
    digits = "".join(ch for ch in s if ch.isdigit() or ch == "-")
    if digits and digits != "-":
        try:
            resseq = int(digits)
            if s and s[-1].isalpha():
                icode = s[-1]
            return resseq, normalize_icode(icode)
        except Exception:
            return None, None

    return None, None


def run_mkdssp_to_file(pdb_path: str, mkdssp_exe: str, out_dssp_path: str) -> None:
    proc = subprocess.run(
        [mkdssp_exe, pdb_path, out_dssp_path],
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    if proc.returncode != 0:
        raise RuntimeError(
            f"mkdssp failed (rc={proc.returncode})\n"
            f"cmd: {mkdssp_exe} {pdb_path} {out_dssp_path}\n"
            f"stderr:\n{proc.stderr}"
        )


def _angle_to_value(x: float, missing_value: float = 0.0) -> float:
    try:
        v = float(x)
    except Exception:
        return missing_value
    if abs(v - 360.0) < 1e-6:
        return missing_value
    return float(v)


def parse_dssp_tuple(dssp_data):
    aa = str(dssp_data[0]).strip()
    ss = str(dssp_data[1]).strip()
    acc = float(dssp_data[2])
    phi = _angle_to_value(dssp_data[3])
    psi = _angle_to_value(dssp_data[4])
    max_acc = float(residue_max_acc.get(aa.upper(), 0.0))
    rasa = acc / max_acc if max_acc > 1e-6 else 0.0
    return aa.upper(), ss, acc, rasa, phi, psi


def compute_dssp_with_chain_asa(
    pdb_path,
    mkdssp_exe,
    probe_radius=1.4,
    n_points=960,
    debug_bad_keys=False,
    keep_tmp_dssp=False,
):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("protein", pdb_path)
    pdb_id = Path(pdb_path).stem

    tmp_dssp = os.path.join(tempfile.gettempdir(), f"{pdb_id}.{os.getpid()}.dssp")
    run_mkdssp_to_file(pdb_path, mkdssp_exe, tmp_dssp)

    dssp_dict, dssp_keys = make_dssp_dict(tmp_dssp)

    if not keep_tmp_dssp and os.path.isfile(tmp_dssp):
        try:
            os.remove(tmp_dssp)
        except Exception:
            pass

    sr_complex = ShrakeRupley(probe_radius=probe_radius, n_points=n_points)
    struct_complex = copy.deepcopy(structure)
    model_complex = struct_complex[0]
    sr_complex.compute(model_complex, level="R")

    asa_complex = {}
    for chain in model_complex:
        chain_id = chain.id
        for residue in chain:
            if not is_aa(residue):
                continue
            resseq, icode = parse_resid_to_resseq_icode(residue.id)
            if resseq is None:
                continue
            asa_complex[(chain_id, resseq, icode)] = float(getattr(residue, "sasa", 0.0))

    rows = []
    for key in dssp_keys:
        chain_id = key[0]
        resid = key[1]
        resseq, icode = parse_resid_to_resseq_icode(resid)
        if resseq is None:
            if debug_bad_keys:
                print(f"[WARN] skip bad DSSP resid: pdb={pdb_id} key={key}", flush=True)
            continue

        dssp_data = dssp_dict[key]
        try:
            aa, ss, acc, rasa, phi, psi = parse_dssp_tuple(dssp_data)
        except Exception as e:
            if debug_bad_keys:
                print(f"[WARN] bad dssp_data: pdb={pdb_id} key={key} err={e}", flush=True)
            continue

        if ss == "H":
            ss_H, ss_E, ss_C = 1.0, 0.0, 0.0
        elif ss == "E":
            ss_H, ss_E, ss_C = 0.0, 1.0, 0.0
        else:
            ss_H, ss_E, ss_C = 0.0, 0.0, 1.0

        asa_c = asa_complex.get((chain_id, resseq, icode), 0.0)

        rows.append(
            {
                "pdb_id": pdb_id,
                "chain_id": chain_id,
                "resseq": resseq,
                "icode": icode,
                "aa": aa,
                "ss_H": ss_H,
                "ss_E": ss_E,
                "ss_C": ss_C,
                "ACC": acc,
                "RASA": rasa,
                "phi": phi,
                "psi": psi,
                "ASA_complex": asa_c,
            }
        )

    df = pd.DataFrame(rows)
    if len(df) > 0:
        df.sort_values(["chain_id", "resseq", "icode"], inplace=True)
    return df


def process_one_pdb(args):
    pdb_path, out_csv, mkdssp_exe, idx, total, skip_existing = args
    pdb_path = Path(pdb_path)
    out_csv = Path(out_csv)

    if skip_existing and out_csv.exists():
        print(f"[SKIP] {idx + 1}/{total} {pdb_path.name} -> {out_csv}", flush=True)
        return

    print(f"[DSSP] {idx + 1}/{total} processing {pdb_path}", flush=True)

    try:
        df = compute_dssp_with_chain_asa(str(pdb_path), mkdssp_exe)
        df.to_csv(out_csv, index=False)
        print(f"[OK] wrote {out_csv}", flush=True)
    except Exception as e:
        print(f"[ERROR] failed {pdb_path}: {e}", flush=True)


def collect_tasks(pdb_dir: Path, out_dir: Path, mkdssp_exe: str, skip_existing: bool):
    pdb_files = sorted(pdb_dir.glob("*.pdb"))
    tasks = []
    total = len(pdb_files)
    for idx, pdb_path in enumerate(pdb_files):
        pdb_id = pdb_path.stem
        out_csv = out_dir / f"{pdb_id}.csv"
        tasks.append((str(pdb_path), str(out_csv), mkdssp_exe, idx, total, skip_existing))
    return tasks


def run_for_one_side(name: str, pdb_dir: Path, out_dir: Path, mkdssp_exe: str, skip_existing: bool, num_workers: int):
    out_dir.mkdir(parents=True, exist_ok=True)

    tasks = collect_tasks(pdb_dir, out_dir, mkdssp_exe, skip_existing)
    total = len(tasks)

    print("=" * 80, flush=True)
    print(f"[{name}] pdb_dir={pdb_dir}", flush=True)
    print(f"[{name}] out_dir={out_dir}", flush=True)
    print(f"[{name}] total pdb files={total}", flush=True)
    print("=" * 80, flush=True)

    if total == 0:
        print(f"[{name}] no pdb files found, skip.", flush=True)
        return

    global _POOL
    actual_workers = max(1, min(num_workers, total))
    _POOL = mp.Pool(processes=actual_workers, initializer=init_worker)
    try:
        _POOL.map(process_one_pdb, tasks)
        _POOL.close()
        _POOL.join()
    except KeyboardInterrupt:
        _POOL.terminate()
        _POOL.join()
        raise

def main():
    wt_pdb_dir = Path(DATA_CONFIG["wt_pdb_dir"])
    mut_pdb_dir = Path(DATA_CONFIG["mut_pdb_dir"])

    wt_out_dir = Path(DSSP_CONFIG["wt_dir"])
    mut_out_dir = Path(DSSP_CONFIG["mut_dir"])

    mkdssp_exe = "mkdssp"
    skip_existing = True

    cpu_count = os.cpu_count() or 8
    num_workers = min(16, max(1, cpu_count // 2))

    print("=" * 80, flush=True)
    print(f"[DSSP] mkdssp_exe={mkdssp_exe}", flush=True)
    print(f"[DSSP] skip_existing={skip_existing}", flush=True)
    print(f"[DSSP] cpu_count={cpu_count}", flush=True)
    print(f"[DSSP] num_workers={num_workers}", flush=True)
    print("=" * 80, flush=True)

    run_for_one_side(
        name="WT",
        pdb_dir=wt_pdb_dir,
        out_dir=wt_out_dir,
        mkdssp_exe=mkdssp_exe,
        skip_existing=skip_existing,
        num_workers=num_workers,
    )

    run_for_one_side(
        name="MUT",
        pdb_dir=mut_pdb_dir,
        out_dir=mut_out_dir,
        mkdssp_exe=mkdssp_exe,
        skip_existing=skip_existing,
        num_workers=num_workers,
    )


if __name__ == "__main__":
    main()