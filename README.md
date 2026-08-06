# Renal CT Research Explorer

A cloud-ready, multi-dataset Streamlit viewer for renal CT research. The application runs
loading, normalization, measurement, windowing, and rendering on the server; a user's device
only receives the rendered interface and compact result tables.

**Live app:** https://renal-ct-explorer-9jxxrehbetm5lbbqffkjui.streamlit.app/

The default **Online demo** mode downloads only the selected public KiTS23 image from the
official Hugging Face repository and its segmentation from the official GitHub repository.
No dataset installation or local path is required for visitors.

This is research software. It is not a medical device and must not be used for diagnosis.

## What this release changes

- Adds adapters for KiTS23, KiPA22, converted C4KC-KiTS19, AbdomenAtlas, AMOS22, and FLARE22.
- Normalizes dataset-specific masks into a shared label vocabulary.
- Uses true KiPA22 renal artery and vein labels instead of claiming that an HU threshold can
  distinguish vessels.
- Validates image/segmentation dimensions and affine matrices before analysis.
- Calculates volumes and distances in physical coordinates.
- Separates kidney and tumor connected components and derives research-only laterality.
- Preserves physical aspect ratios in coronal and sagittal views.
- Builds rotatable physical-coordinate 3D surfaces with isotropic signed-distance
  reconstruction, volume-calibrated contouring, non-shrinking smoothing, topology-preserving
  decimation, and an on-screen mesh quality report.
- Adds an explicitly experimental Frangi/Sato connectivity-filtered candidate-vessel surface
  and centerline layer for datasets such as KiTS23 that do not provide vessel annotations.
- Keeps CT volumes in a shared server cache rather than Streamlit session state.
- Exports JSON and CSV through browser downloads.
- Provides a container configuration for online deployment.
- Provides an on-demand, three-case KiTS23 catalogue for free Streamlit hosting.
- Limits the in-memory case cache to one entry for free-tier stability.

The **Interactive 3D** tab is generated on demand. True renal artery and vein surfaces are
shown when a dataset supplies those labels (for example KiPA22). KiTS23 does not include
renal vessel annotations, so its optional cyan vessel layer is an image-derived research
candidate only and does not classify arteries versus veins.

## 3D quality architecture

The 3D pipeline is designed around measurable geometric fidelity, not only a smooth-looking
screenshot:

1. Each label is cropped with a physical-space border and converted to a signed-distance
   field using the NIfTI voxel spacing.
2. The field is resampled isotropically. Small structures automatically receive a finer
   target grid so lesions and cysts are not sacrificed for whole-organ speed.
3. Lewiner marching cubes extracts a topologically correct surface. The iso-level is
   calibrated in millimetres against the source label volume before smoothing.
4. VTK windowed-sinc smoothing uses the same non-shrinking family and parameter mapping as
   3D Slicer's closed-surface conversion. Smoothing is reduced automatically if its 95th
   percentile displacement or volume error exceeds the selected quality profile.
5. Topology-preserving decimation is accepted only when the surface remains watertight and
   does not worsen volume error by more than one percentage point.
6. The app reports triangles, source-to-mesh volume error, surface area, smoothing
   displacement, mesh spacing, and watertight status for every generated structure.

For unlabeled vessels, the app combines physical-scale Frangi and Sato responses, dual
threshold connectivity, renal-proximity and component filtering, a volume-calibrated mesh,
and a skeleton centerline. This is a stronger visualization heuristic than a single global
Frangi threshold, but it remains a candidate layer—not an artery/vein segmentation.

### Release quality gates

For the recommended **Balanced** profile, labeled structures must have zero boundary,
non-manifold, and degenerate edges; organ/tumor volume error should remain at or below 3%;
small structures may use up to 6%; and smoothing displacement p95 must remain at or below
0.9 mm. Experimental vessel candidates use an 8% volume-fidelity limit because the source
mask itself is image-derived.

On public KiTS23 `case_00015`, the current Balanced build produced these reproducible smoke
benchmark results on the development machine:

| Structure | Triangles | Volume error | Smooth p95 | Watertight |
|---|---:|---:|---:|:---:|
| Kidney | 120,000 | 0.02% | 0.53 mm | Yes |
| Tumor | 10,678 | 1.14% | 0.57 mm | Yes |
| Cyst | 6,716 | 0.13% | 0.43 mm | Yes |
| Candidate vessels | 7,352 | 7.24% | 0.55 mm | Yes |

The labeled surfaces completed in about 2.0 seconds and the full build including the
CPU-intensive experimental vessel pass in about 15.6 seconds. Hosting hardware and case size
will change those timings. The previous coarse implementation had 15.04% cyst volume error on
the same case; the new result is 0.13%.

These checks support a defensible quality claim for this web workflow, not a universal claim
that it outperforms every desktop medical workstation or a clinical validation study. A fair
competitor comparison must use the same source labelmaps, target triangle budgets, and
surface-distance/volume/topology metrics.

### Public methods studied

- [3D Slicer closed-surface conversion](https://github.com/Slicer/Slicer/blob/c3467ecd3f8d617b2ef8b6ee469c6da2436c748c/Libs/vtkSegmentationCore/vtkBinaryLabelmapToClosedSurfaceConversionRule.cxx)
- [VTK windowed-sinc smoothing](https://vtk.org/doc/nightly/html/classvtkWindowedSincPolyDataFilter.html)
- [VTK discrete flying edges](https://vtk.org/doc/nightly/html/classvtkDiscreteFlyingEdges3D.html)
- [VTK SurfaceNets3D](https://vtk.org/doc/nightly/html/classvtkSurfaceNets3D.html)
- [scikit-image Lewiner marching cubes](https://scikit-image.org/docs/stable/api/skimage.measure.html)
- [VMTK vascular modeling toolkit](https://vmtk.github.io/documentation/getting-started.html)

## Supported labels

| Canonical label | KiTS23 | KiPA22 | C4KC-KiTS | AbdomenAtlas | AMOS22 | FLARE22 |
|---|---:|---:|---:|---:|---:|---:|
| Kidney | 1 | 2 | 1 | 3, 4 | 2, 3 | 2, 13 |
| Tumor | 2 | 4 | 2 | — | — | — |
| Cyst | 3 | — | — | — | — | — |
| Renal artery | — | 3 | — | — | — | — |
| Renal vein | — | 1 | — | — | — | — |
| Aorta | — | — | — | 1 | 8 | 5 |
| Inferior vena cava | — | — | — | 7 | 9 | 6 |

Dataset labels are mapped at load time. Unsupported source labels remain background; the
original values discovered in each mask are recorded in the case report.

## Run locally

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
export RENAL_DATA_ROOT=/absolute/path/to/datasets
streamlit run app.py
```

Without `RENAL_DATA_ROOT`, the application starts in Online demo mode. With
`RENAL_DATA_ROOT` configured, it starts in Server datasets mode while still allowing users to
switch to the online demonstration.

`RENAL_DATA_ROOT` should contain one subfolder per dataset, for example:

```text
/data
├── kits23
│   └── case_00000
│       ├── imaging.nii.gz
│       └── segmentation.nii.gz
├── kipa22
│   └── case_00000
│       ├── imaging.nii.gz
│       └── segmentation.nii.gz
├── abdomen_atlas
│   └── AbdomenAtlas_case
│       ├── ct.nii.gz
│       └── combined_labels.nii.gz
└── amos22
    ├── imagesTr
    └── labelsTr
```

The dataset folder can also be changed from the sidebar. It is a server path—not a path on
the viewer's computer once the application is deployed.

## Obtain KiTS23

Use the official project. Its downloader retrieves imaging from the official Hugging Face
dataset and places it alongside the versioned segmentations.

```bash
git clone https://github.com/neheller/kits23
cd kits23
pip install -e .
kits23_download_data
```

Move or mount the resulting `dataset` folder as `/data/kits23` in the hosted application.

KiPA22 requires joining the official Grand Challenge. C4KC-KiTS is available through TCIA;
this release expects selected DICOM series and segmentations to be converted into aligned
NIfTI files. Raw DICOM and DICOM-SEG ingestion is intentionally not claimed yet.

## Run online with Docker

Set the host location of your persistent data and start the service:

```bash
export RENAL_DATA_HOST_PATH=/absolute/path/to/datasets
docker compose up --build
```

Open `http://localhost:8501`. On a hosted machine, expose port 8501 behind HTTPS and an
authentication layer. The container mounts the data read-only.

Do not bake datasets into the container image. Keep them on a persistent disk or object-store
cache so deployments do not repeatedly download hundreds of gigabytes.

## Deploy free on Streamlit Community Cloud

1. Push this folder to a GitHub repository.
2. Open <https://share.streamlit.io/> and select **Create app**.
3. Choose the repository and its default branch.
4. Set the entrypoint to `app.py`.
5. Deploy without adding secrets; the public KiTS23 demo uses only public official URLs.

Do not add any CT volumes to the Git repository. Streamlit downloads the selected case into
temporary server storage and reuses that file while the application instance remains alive.
The download may repeat after Streamlit restarts or redeploys the app.

The free deployment is intended as a public research demonstration. The three-case catalogue
and one-case memory cache are deliberate safeguards. A full-dataset or multi-user production
deployment needs persistent object storage and larger memory limits.

## Cloud architecture

This first release is deliberately a single hosted service:

```text
Browser -> hosted Streamlit process -> read-only persistent dataset volume
                              \-> shared in-memory case/analysis cache
```

That already removes computation from the user's computer. The next scaling milestone is to
move `load_case_bundle` behind an analysis API and background job queue, storing normalized
volumes and reports in object storage. This is needed for multiple application replicas,
long-running model inference, or GPU workers; it is unnecessary complexity for a small
research deployment.

## Dataset sources and terms

- KiTS23: <https://github.com/neheller/kits23> — CC BY-NC-SA 4.0 data.
- KiPA22: <https://kipa22.grand-challenge.org/> — registration and challenge terms apply.
- C4KC-KiTS: <https://www.cancerimagingarchive.net/collection/c4kc-kits/> — CC BY 3.0.
- AbdomenAtlas: <https://github.com/MrGiovanni/AbdomenAtlas> — check the selected release.
- AMOS22: <https://amos22.grand-challenge.org/> — check challenge terms.
- FLARE22: <https://flare22.grand-challenge.org/> — check the selected release.

Keep dataset attribution and license information attached to every deployment and export.
Do not assume that the most permissive license applies when multiple datasets are combined.

## Verify

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```
