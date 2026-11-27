# Private planner on a Mac, reachable from anywhere

The public Streamlit Cloud app stays a demo with public KiTS23 cases. Patient CTs are processed
only on your own Apple Silicon Mac, in **patient mode**. You and colleagues reach that Mac
through an authenticated tunnel.

```text
Phone / iPad / hospital PC ──HTTPS + login──▶ tunnel ──▶ Mac: Streamlit (127.0.0.1:8501)
                                                              ├─ background worker (one job at a time)
                                                              ├─ ~/RenalPlanner/cases   (pseudonymised NIfTI)
                                                              ├─ ~/RenalPlanner/inbox   (DICOM drop folder)
                                                              └─ ~/RenalPlanner/models  (weights)
```

> Research software, not a medical device. Before using real patient data, agree the setup with
> your data-protection officer / ethics committee (DSGVO/GDPR). See "Data protection" below.

## 1. Install (once)

```bash
git clone https://github.com/ahmederaky2-cmyk/renal-ct-explorer.git
cd renal-ct-explorer
git checkout planning-engine          # until merged
scripts/mac/setup.sh --with-atlasnet  # ~20–40 min; downloads ~5 GB of model weights
```

The script creates a Python 3.13 environment, the `~/RenalPlanner` folders, downloads
TotalSegmentator-MLX and (optionally) AbdomenAtlasNet-Tumors, and writes
`~/RenalPlanner/planner.env`. **It prints the app password once.** Save it in your password
manager.

Check the install:

```bash
.venv/bin/pytest            # all tests, including the end-to-end pipeline test
scripts/mac/run.sh          # then open http://127.0.0.1:8501
```

To start automatically at login: `scripts/mac/install-launchd.sh`.
In **System Settings → Energy**, enable *Prevent automatic sleeping when the display is off*.
`run.sh` also keeps the Mac awake with `caffeinate` while the planner runs on power.

## 2. Remote access

Pick one. Both give HTTPS without opening ports on your router.

### Option A — Cloudflare Tunnel + Access (browser only, colleagues need nothing installed)

Needs a domain managed by Cloudflare (a cheap domain is enough).

1. Cloudflare dashboard → **Zero Trust → Networks → Tunnels → Create tunnel** (cloudflared).
   Install the connector on the Mac with the command it shows
   (`brew install cloudflared`, then `sudo cloudflared service install <token>`).
2. In the tunnel, add a **public hostname**, e.g. `planner.example.org` → `http://localhost:8501`.
3. **Zero Trust → Access → Applications → Add → Self-hosted** for `planner.example.org`.
   Policy: *Allow*, *Emails* = you and your colleagues. Login method: *One-time PIN*.
   Session duration: e.g. 12 hours.
4. Keep the app password set as a second layer.

**Upload limit:** Cloudflare's Free/Pro plans reject request bodies over **100 MB**, so large
DICOM ZIPs uploaded through the browser will fail. Use the **inbox**: put the ZIP or DICOM
folder into `~/RenalPlanner/inbox` on the Mac (PACS export, USB stick, or a synced network
share), then press *Import from inbox* in the app from any device.

### Option B — Tailscale (end-to-end encrypted, no upload limit)

1. Install Tailscale on the Mac and sign in; run `tailscale serve --bg 8501`.
2. Share the Mac with colleagues (Tailscale admin console → *Share*); they install Tailscale.
3. Open `https://<mac-name>.<tailnet>.ts.net` from any of your devices.

Traffic is encrypted end to end between devices, so no third party sees decrypted data.
This is usually the easier conversation with a data-protection officer.

## 3. Daily workflow

1. **Patient cases → Upload ZIP** (≤ 100 MB through Cloudflare) **or Import from inbox**.
   Give only a short label: never a name, date of birth or hospital number. An inbox study
   is deleted only after its DICOM → NIfTI conversion has succeeded; if the import or
   conversion fails, it stays in the inbox.
2. Check the series table. Each phase is guessed from the series description (English and
   German keywords: *arteriell, portalvenös, Spätphase, nativ …*). Correct it if needed.
   The nephrographic/portal-venous series becomes the planning grid.
3. **Start automatic segmentation.** The job runs in the background, so you can close the
   browser. Typical steps:
   - DICOM → NIfTI, then the raw DICOM is deleted.
   - TotalSegmentator (MLX on Metal, else PyTorch/MPS): kidneys, aorta, IVC, adrenals, liver,
     spleen, pancreas, duodenum, colon, psoas.
   - TotalSegmentator `kidney_cysts` subtask (PyTorch).
   - AbdomenAtlasNet-Tumors (nnU-Net v2): kidney tumours.
   - Arterial and excretory phases are registered rigidly onto the planning grid
     (renal region of interest).
   - Excretory phase → collecting system (contrast-filled urine inside the renal envelope).
   - Arterial phase → renal artery/vein, if a validated KiPA22 ONNX model is installed.
4. Open the case. **Review every label.** The *Review & correct labels* panel shows warnings,
   offers the CT + labels as NIfTI and a 3D Slicer colour table. Correct the labels in Slicer
   and upload them; the previous version is kept in `history/`.
5. Use the **Surgical plan** tab: R.E.N.A.L./PADUA, distances (including duodenum, colon,
   pancreas, spleen), resection margin, territories, PDF/STL export.

## 4. Data protection built in

- Raw DICOM is deleted right after conversion (also when it fails, and when a worker dies).
  NIfTI files contain no DICOM header fields. Series descriptions and UIDs, which can hold
  free text, are kept only beside the raw upload and deleted with it; saved requests use
  opaque handles (S1, S2 …).
- Cases get random IDs (`PN-xxxxxxxx`) and are deleted automatically after
  `RENAL_RETENTION_DAYS` (default 30). Cases can be deleted manually at any time.
- Case folders are only readable by your macOS user (`chmod 700`).
- Streamlit listens on `127.0.0.1` only; access goes through the tunnel and its login.
- Turn on **FileVault** (full-disk encryption) on the Mac.
- Keep the mapping between `PN-` IDs and patients outside this system, or don't keep one.
- Clinician reviews are stored in `~/RenalPlanner/reviews` (`chmod 700`). They hold the
  reviewer's name, checks and notes, keyed by content hashes; they are not deleted with the
  case, so keep patient details out of review notes.

What still needs institutional sign-off: processing patient data on a personal device, any
third-party tunnel (Cloudflare decrypts traffic at its edge; Tailscale does not), and
whether CT volumes count as pseudonymised or identifiable (faces are not in an abdominal CT,
but body habitus can be distinctive).

## 5. Model licences

| Component | Licence | Notes |
|---|---|---|
| TotalSegmentator code + open-task weights | Apache-2.0 | Not intended for clinical use (authors) |
| totalsegmentator-mlx (pinned 0.1.x) | Apache-2.0 code; upstream weights | Research and education. 0.4 removed the API used here; the pipeline falls back to PyTorch |
| AbdomenAtlasNet-Tumors | **CC BY-NC 4.0** | Non-commercial research only |
| KiPA22-trained vessel model | KiPA22 terms (CC BY-NC-ND) | Train it yourself; see `training/README.md` |

## 6. Settings (`~/RenalPlanner/planner.env`)

| Variable | Default | Meaning |
|---|---|---|
| `RENAL_TOTALSEG_BACKEND` | `auto` | `mlx`, `pytorch` or `auto` (MLX if installed) |
| `RENAL_TOTALSEG_FAST` | `0` | `1` = 3 mm model (faster, coarser) |
| `RENAL_DEVICE` | `auto` | PyTorch device: `mps`, `cpu`, `gpu` |
| `RENAL_RUN_KIDNEY_CYSTS` | `1` | Run the cyst subtask |
| `RENAL_ATLASNET_FOLDS` | `all` | e.g. `0` for one fold (faster) |
| `RENAL_CS_THRESHOLD_HU` | `400` | Excretory-phase urine threshold |
| `RENAL_RETENTION_DAYS` | `30` | Automatic deletion; `0` disables |
| `RENAL_BIND_ADDRESS` | `127.0.0.1` | `0.0.0.0` to also allow direct LAN access |
| `RENAL_REVIEW_DIR` | `~/RenalPlanner/reviews` | Clinician review records (one JSON per exact CT + labels) |

## 7. Train your own collecting-system model

Once you have corrected cases with an excretory phase, export them for nnU-Net:

```bash
.venv/bin/python scripts/export_nnunet_dataset.py --dataset-id 501 \
    --labels KIDNEY TUMOR CYST COLLECTING_SYSTEM --corrected-only \
    --output ~/RenalPlanner/nnUNet_raw
```

Train on an NVIDIA GPU, a cloud GPU for example (`nnUNetv2_plan_and_preprocess -d 501`, then
`nnUNetv2_train 501 3d_fullres 0..4`). Place the trained model folder under `~/RenalPlanner/models`.
This requires ethics approval for research use of the scans.
