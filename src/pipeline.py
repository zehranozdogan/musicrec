#!/usr/bin/env python3
"""MTG-Jamendo için tar-tar embedding hattı (TAM ŞARKI, pencere bazlı çıktı, çoklu model).

Her şarkı 30 sn'lik pencerelere bölünür (şarkının tamamı örtülür, en fazla --max-windows pencere).
--models ile seçilen modeller (panns, vggish, mert, clap) aynı geçişte çalışır, böylece indirme ve
ses çözme maliyeti paylaşılır. Çıktı pencere bazlıdır (sonradan pencereler üzerinde attention gibi
modeller eğitilebilir); ortalama alma eğitim tarafında (train.py) yapılır.
Her tar için: indir (bir sonraki tar arka planda iner) -> doğrula -> aç -> embedding -> .npz yaz -> sesi sil.
Disk hiç dolmaz. Çıktıda var olan tar'lar atlanır, böylece Kaggle'da oturum/süre dolsa bile
kaldığı yerden devam edilir (--prev-dir ile önceki çıktıyı ver). Sonradan yeni bir model eklemek için
aynı --prev-dir ile yalnızca o modeli (--models mert) çalıştırman yeter.

Çıktı düzeni (--out-dir):
    <model>/<tar_adı>.npz   ids (N,), n_win (N,) şarkı başına pencere sayısı,
                            emb (toplam_pencere, ...) float16:
                            panns (W, 2048), vggish (W, 256: ortalama+std), mert (W, 13, 768), clap (W, 512),
                            mel (W, 64, 938: sıfırdan CNN için log-mel, 16 kHz)
    splits/{train,validation,test}.tsv, vocab.json, config.json, failed.txt
"""
import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from common import DATASET, SPLITS, WIN_SEC, TrackDataset, build_vocab, read_split, resample_windows
from extract_panns import PannsEmbedder
from extract_vggish import VggishEmbedder

URLS = {
    "mtg-fast": "https://cdn.freesound.org/mtg-jamendo/{dataset}/{dtype}/{name}",
    "mtg": "https://essentia.upf.edu/documentation/datasets/mtg-jamendo/{dataset}/{dtype}/{name}",
    # freesound CDN kapalıyken: şarkılar Jamendo'dan tek tek (MTG track id = Jamendo track id, 96 kbps mp3)
    "jamendo": "https://prod-1.storage.jamendo.com/?trackid={tid}&format=mp31",
}


KNOWN_MODELS = ("panns", "vggish", "mert", "clap", "mel")
FORMAT_VERSION = 2


def build_embedders(models, device, args) -> dict:
    d = {}
    for m in models:
        if m == "panns":
            d[m] = PannsEmbedder(device, args.panns_batch)
        elif m == "vggish":
            d[m] = VggishEmbedder(device)
        elif m == "mert":
            from extract_mert import MertEmbedder
            d[m] = MertEmbedder(device, args.mert_batch)
        elif m == "clap":
            from extract_clap import ClapEmbedder
            d[m] = ClapEmbedder(device, args.clap_batch, args.clap_name)
        elif m == "mel":
            from extract_mel import MelEmbedder
            d[m] = MelEmbedder(device)
    return d


def ensure_splits(mtg_dir: Path, split_idx: int, dataset: str) -> dict:
    d = mtg_dir / "data" / "splits" / f"split-{split_idx}"
    files = {s: d / f"{dataset}-{s}.tsv" for s in SPLITS}
    if not all(p.exists() for p in files.values()):
        print("Split dosyaları yok, split_filter_subset.py ile üretiliyor...")
        subprocess.run(
            [sys.executable, "scripts/split_filter_subset.py", "data/splits",
             "autotagging", dataset, "--category", "mood/theme"],
            cwd=mtg_dir, check=True,
        )
    missing = [str(p) for p in files.values() if not p.exists()]
    if missing:
        listing = sorted(x.name for x in d.glob("*")) if d.exists() else "klasör yok"
        raise SystemExit(f"Split dosyaları bulunamadı: {missing}\n{d} içeriği: {listing}")
    return files


def read_tar_list(mtg_dir: Path, dataset: str, dtype: str) -> list:
    f = mtg_dir / "data" / "download" / f"{dataset}_{dtype}_sha256_tars.txt"
    items = []
    for line in f.read_text().splitlines():
        parts = line.split()
        if len(parts) >= 2:
            items.append((parts[1], parts[0]))  # (dosya adı, sha256)
    if not items:
        raise SystemExit(f"Tar listesi boş: {f}")
    return items


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download(url: str, dest: Path, sha: str, retries: int = 8) -> None:
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as r:
                r.raise_for_status()
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
            if sha256_of(tmp) != sha:
                raise RuntimeError("sha256 uyuşmuyor")
            os.replace(tmp, dest)
            return
        except Exception as e:  # noqa: BLE001
            print(f"  indirme hatası ({attempt}/{retries}): {e}")
            tmp.unlink(missing_ok=True)
            if attempt == retries:
                raise
            time.sleep(min(30 * attempt, 180))  # sunucu 503 verirse dakikalarca bekle


def fetch_jamendo_folder(folder: str, rows: dict, dest: Path, workers: int = 8):
    """Tar klasörüne (ör. "00") düşen şarkıları Jamendo'dan tek tek indirir. Döner: (dest, inmeyen id'ler)."""
    dest.mkdir(parents=True, exist_ok=True)
    tids = sorted(t for t, r in rows.items() if r["path"].split("/")[0] == folder)
    if not tids:
        raise RuntimeError(f"klasör {folder} için split'lerde şarkı yok (path biçimi beklenenden farklı?)")

    def one(tid):
        f = dest / f"{tid}.mp3"
        if f.exists():
            return None
        for attempt in range(1, 4):
            try:
                with requests.get(URLS["jamendo"].format(tid=tid), stream=True, timeout=60) as r:
                    if r.status_code == 404:
                        return tid
                    r.raise_for_status()
                    if "audio" not in r.headers.get("content-type", ""):
                        raise RuntimeError(f"ses değil: {r.headers.get('content-type')}")
                    tmp = f.with_suffix(".part")
                    with open(tmp, "wb") as fh:
                        for chunk in r.iter_content(1 << 20):
                            fh.write(chunk)
                if tmp.stat().st_size < 20_000:
                    tmp.unlink(missing_ok=True)
                    raise RuntimeError("dosya çok küçük")
                os.replace(tmp, f)
                return None
            except Exception:  # noqa: BLE001
                time.sleep(5 * attempt)
        return tid

    with ThreadPoolExecutor(max_workers=workers) as ex:
        missing = [t for t in ex.map(one, tids) if t is not None]
    if len(missing) > 0.5 * max(len(tids), 1):
        raise RuntimeError(f"klasör {folder}: {len(missing)}/{len(tids)} şarkı Jamendo'dan inmedi")
    return dest, missing


def safe_extract(tar_path: Path, dest: Path) -> None:
    """Yol kaçışı ve link içeren üyeleri atlayarak açar."""
    dest = dest.resolve()
    with tarfile.open(tar_path) as tar:
        members = []
        for m in tar.getmembers():
            target = (dest / m.name).resolve()
            if m.issym() or m.islnk() or not str(target).startswith(str(dest)):
                continue
            members.append(m)
        tar.extractall(dest, members=members)


def save_npz(path: Path, ids, n_win, embs) -> None:
    """embs: şarkı başına (n_pencere, ...) dizilerinin listesi; pencere ekseninde birleştirilip float16 yazılır."""
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez(tmp, ids=np.asarray(ids, dtype=np.int64), n_win=np.asarray(n_win, dtype=np.int32),
             emb=np.concatenate(embs).astype(np.float16))
    os.replace(tmp, path)


def check_config(prev_dir, out: Path, cfg: dict) -> None:
    """Önceki çıktı farklı ayarlarla üretildiyse karıştırmamak için durur."""
    if prev_dir:
        pc = Path(prev_dir)
        found = list(pc.rglob("config.json"))
        if not found:
            raise SystemExit(f"{prev_dir} içinde config.json yok: eski (30 sn kesit) çıktı olabilir, karıştırılamaz.")
        old = json.loads(found[0].read_text())
        if old != cfg:
            raise SystemExit(f"Önceki çıktının ayarları farklı, devam edilemez.\n  önceki: {old}\n  şimdiki: {cfg}")
    (out / "config.json").write_text(json.dumps(cfg, indent=1))


def restore_prev(prev_dir, out: Path, skip=()) -> None:
    if not prev_dir:
        return
    n = 0
    for f in Path(prev_dir).rglob("*.npz"):
        if f.parent.name in KNOWN_MODELS and f.parent.name not in skip:
            dst = out / f.parent.name / f.name
            if not dst.exists():
                dst.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(f, dst)
                n += 1
    print(f"Önceki çıktıdan {n} dosya geri yüklendi.")


def process_audio(audio_dir: Path, rows: dict, embedders: dict, args):
    """Döner: ids, şarkı başına pencere sayıları, feats[model] (şarkı başına (n_pencere, ...) listesi),
    hatalı id'ler, toplam şarkı sayısı, toplam pencere sayısı."""
    items = []
    for p in sorted(audio_dir.rglob("*.mp3")):
        try:
            tid = int(p.name.split(".")[0])
        except ValueError:
            continue
        if tid in rows:
            items.append((tid, p, rows[tid]["duration"]))
    loader = DataLoader(
        TrackDataset(items, args.max_windows), batch_size=None, num_workers=args.num_workers,
        shuffle=False, prefetch_factor=2 if args.num_workers else None,
    )
    ids, nwins, failed, total_win = [], [], [], 0
    feats = {m: [] for m in embedders}
    for tid, w32, err in tqdm(loader, desc="  embedding", leave=False):
        if err is not None:
            failed.append(tid)
            continue
        try:
            w32 = w32.numpy()
            resampled, per = {}, {}
            for m, emb in embedders.items():
                if emb.sr not in resampled:
                    resampled[emb.sr] = resample_windows(w32, emb.sr)
                per[m] = emb.windows(resampled[emb.sr])
        except Exception as e:  # noqa: BLE001
            print(f"  {tid}: embedding hatası {type(e).__name__}: {e}")
            failed.append(tid)
            if not ids and len(failed) >= 8:  # ilk 8 şarkının hepsi patladı: sistematik hata, hemen dur
                raise RuntimeError(f"ilk {len(failed)} şarkının hepsi embedding hatası verdi, tar iptal: {e}") from e
            continue
        ids.append(tid)
        nwins.append(len(w32))
        for m in embedders:
            feats[m].append(per[m])
        total_win += len(w32)
    return ids, nwins, feats, failed, len(items), total_win


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mtg-dir", required=True, help="klonlanmış mtg-jamendo-dataset reposu")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prev-dir", default=None, help="önceki turun çıktısı (devam için)")
    default_work = "/kaggle/temp" if Path("/kaggle/temp").exists() else "work"
    ap.add_argument("--work-dir", default=default_work)
    ap.add_argument("--dataset", default=DATASET)
    ap.add_argument("--type", default="audio-low", dest="dtype")
    ap.add_argument("--source", default="mtg-fast", choices=list(URLS))
    ap.add_argument("--split", type=int, default=0)
    ap.add_argument("--max-hours", type=float, default=10.5, help="süre bütçesi; dolunca temiz çıkar")
    ap.add_argument("--panns-batch", type=int, default=16, help="PANNs'a aynı anda verilen pencere sayısı")
    ap.add_argument("--models", nargs="+", default=["panns", "vggish"], choices=list(KNOWN_MODELS),
                    help="aynı geçişte çalışacak modeller")
    ap.add_argument("--mert-batch", type=int, default=12, help="MERT'e aynı anda verilen 5 sn'lik parça sayısı")
    ap.add_argument("--clap-batch", type=int, default=24, help="CLAP'a aynı anda verilen 10 sn'lik parça sayısı")
    ap.add_argument("--clap-name", default="music_audioset_epoch_15_esc_90.14", help="resmi LAION CLAP checkpoint adı")
    ap.add_argument("--redo", nargs="+", default=[], choices=list(KNOWN_MODELS),
                    help="--prev-dir'den bu modelleri geri yükleme, yeniden çıkar (ör. --redo clap)")
    ap.add_argument("--max-windows", type=int, default=20, help="şarkı başına en fazla pencere (20 = 10 dk)")
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--limit-tars", type=int, default=0, help="test için ilk N tar")
    args = ap.parse_args()

    t0 = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Cihaz:", device)
    mtg, out, work = Path(args.mtg_dir), Path(args.out_dir), Path(args.work_dir) / "musicrec"
    for m in (*args.models, "splits"):
        (out / m).mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    cfg = {"format": FORMAT_VERSION, "win_sec": WIN_SEC, "max_windows": args.max_windows}
    check_config(args.prev_dir, out, cfg)
    restore_prev(args.prev_dir, out, set(args.redo))

    files = ensure_splits(mtg, args.split, args.dataset)
    rows_by_split = {s: read_split(p) for s, p in files.items()}
    rows = {r["track_id"]: r for rs in rows_by_split.values() for r in rs}
    for s, p in files.items():
        shutil.copy2(p, out / "splits" / f"{s}.tsv")
    vocab = build_vocab(rows_by_split)
    (out / "vocab.json").write_text(json.dumps(vocab, ensure_ascii=False, indent=1))
    print(f"{len(rows)} şarkı, {len(vocab)} etiket.")

    tars = read_tar_list(mtg, args.dataset, args.dtype)
    if args.limit_tars:
        tars = tars[: args.limit_tars]

    embedders = build_embedders(args.models, device, args)
    last, finished, n_ok, n_fail = 0.0, 0, 0, 0

    def fetch(name, sha):
        if args.source == "jamendo":
            folder = name.removesuffix(".tar").rsplit("-", 1)[1]
            return fetch_jamendo_folder(folder, rows, work / f"audio_{name.removesuffix('.tar')}")
        path = work / name
        if not path.exists():
            url = URLS[args.source].format(dataset=args.dataset, dtype=args.dtype, name=name)
            download(url, path, sha)
        return path

    todo = [(n, h) for n, h in tars if not all((out / m / f"{n.removesuffix('.tar')}.npz").exists()
                                                 for m in args.models)]
    finished = len(tars) - len(todo)
    print(f"{finished} tar zaten hazır, {len(todo)} tar işlenecek.")
    pool = ThreadPoolExecutor(max_workers=1)
    fut = pool.submit(fetch, *todo[0]) if todo else None
    try:
        for k, (name, sha) in enumerate(todo):
            stem = name.removesuffix(".tar")
            if k > 0 and time.time() - t0 + last * 1.1 > args.max_hours * 3600:
                print("Süre bütçesi doldu, temiz çıkılıyor. Kalan tar'lar için bu çıktıyı --prev-dir ile ver.")
                break
            ts = time.time()
            audio_dir = work / f"audio_{stem}"
            tar_path = work / name
            try:
                print(f"[{finished + 1}/{len(tars)}] {name}: indirme bekleniyor...")
                got = fut.result()
                if k + 1 < len(todo):
                    fut = pool.submit(fetch, *todo[k + 1])  # sonraki tar arka planda iner
                if args.source == "jamendo":
                    _, missing = got
                    if missing:
                        print(f"  Jamendo'da bulunamayan {len(missing)} şarkı atlandı")
                        with open(out / "missing_jamendo.txt", "a") as f:
                            f.write("\n".join(map(str, missing)) + "\n")
                else:
                    tar_path = got
                    audio_dir.mkdir(exist_ok=True)
                    safe_extract(tar_path, audio_dir)
                    tar_path.unlink(missing_ok=True)
                need = {m: e for m, e in embedders.items() if not (out / m / f"{stem}.npz").exists()}
                ids, nwins, feats, failed, total, nwin = process_audio(audio_dir, rows, need, args)
                if not ids:
                    raise RuntimeError("bu tar'dan hiç embedding çıkmadı")
                for m in need:
                    save_npz(out / m / f"{stem}.npz", ids, nwins, feats[m])
                if failed:
                    with open(out / "failed.txt", "a") as f:
                        f.write("\n".join(map(str, failed)) + "\n")
                n_ok += len(ids)
                n_fail += len(failed)
                finished += 1
                last = time.time() - ts
                left = len(todo) - k - 1
                print(f"  {len(ids)}/{total} şarkı, {nwin} pencere ({nwin / max(len(ids), 1):.1f}/şarkı), "
                      f"{len(failed)} hata, {last / 60:.1f} dk, toplam geçen {(time.time() - t0) / 3600:.2f} sa, "
                      f"kalan tahmini {left * last / 3600:.1f} sa")
                if n_fail > 0.2 * (n_ok + n_fail) and n_ok + n_fail >= 100:
                    raise SystemExit(f"Hata oranı %20'yi aştı ({n_fail}/{n_ok + n_fail}), durduruldu. "
                                     "ffmpeg/librosa kurulumunu kontrol et.")
            finally:
                tar_path.unlink(missing_ok=True)
                shutil.rmtree(audio_dir, ignore_errors=True)
    finally:
        pool.shutdown(wait=False, cancel_futures=True)

    print(f"Bitti: {finished}/{len(tars)} tar tamam.")
    if finished == len(tars):
        print("TAMAM: tüm embedding'ler hazır.")


if __name__ == "__main__":
    main()
