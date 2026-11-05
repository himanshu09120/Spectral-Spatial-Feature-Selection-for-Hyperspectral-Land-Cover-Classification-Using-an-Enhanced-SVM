# WHU-Hi-LongKou data

The only source used is the [official Wuhan University resource page](https://rsidea.whu.edu.cn/e-resource_WHUHi_sharing.htm). Its MATLAB link is [the download questionnaire](https://rsidea-whu-hi.mikecrm.com/MqhjRPN). On 29 October 2026, that form's published redirect configuration contained:

`http://rsidea.whu.edu.cn/code_data/WHU-Hi/Matlab_data_format.zip`

The corresponding HTTPS URL was checked, returned HTTP 200 and advertised 900,809,518 bytes and HTTP range support. This is a discovered official URL, not a guessed mirror. The downloader uses HTTPS and retrieves the ZIP central directory and only the two LongKou entries. It does not submit personal information or a fabricated registration.

```sh
python3 scripts/download_data.py
```

Requirements: Python 3.9+ with standard-library HTTPS support and an internet connection. The official server rejected Python's default user-agent; the downloader uses a browser user-agent. Downloads are streamed into `.part` files, checked against ZIP CRC32 and size, hashed with SHA256, then renamed. Existing destination files are verified and skipped if they match the official archive; mismatches are never overwritten. On retry, completed entries are preserved while incomplete compressed transfers restart rather than resume.

If range access is unavailable, open the official questionnaire in a browser and use the official MATLAB ZIP download. Extract **only**:

```text
Matlab_data_format/WHU-Hi-LongKou/WHU_Hi_LongKou.mat
Matlab_data_format/WHU-Hi-LongKou/WHU_Hi_LongKou_gt.mat
```

Place these directly in `data/raw/`. Do not use the archive's preselected Train/Test maps: this project creates its own documented splits. Do not substitute another scene or combine a cube and labels with inconsistent orientation.

The recorded dataset inspection found variables `WHU_Hi_LongKou` (550×400×270) and `WHU_Hi_LongKou_gt` (550×400). There are 204,542 labelled and 15,458 unlabelled pixels. All class counts match the official table. For a future inspection, `.venv/bin/python -m src.main --validate-data` validates the files without training. The loader converts scene measurements to float32 after validation; labels become int16.

`DATASET_VALIDATION.json` records the inspection. `DOWNLOAD_PROVENANCE.json` records source URLs, actual archive entries, sizes and SHA256 hashes; it is an integrity record, not a signed publisher checksum. Raw files and the downloader's local manifest are excluded from Git. Use the dataset according to the publisher's terms and cite the original authors; this repository does not grant a new dataset license.

## Use in the completed study

The five-seed research run completed on a separate Windows device using the dataset hashes recorded in [the experiment manifest](../report/evidence/manifest.json). Its [saved validation](../report/evidence/data_validation.json) and [protocol preflight](../report/evidence/protocol_feasibility.json) are committed with the report. Every pixel split had 143,179 training-pool, 30,681 validation and 30,682 test pixels; E1–E6 shared a capped fitting subset of 9,000 pixels. All five pixel splits were feasible. The block protocol lacked validation classes for seeds 7, 19 and 73, and no block classifiers were trained.

The 5 November documentation update uses those saved records. Dataset files alone do not reproduce the original outputs: split arrays, predictions and fitted models are stored separately in the experiment device's ignored `outputs/` directory. Read the committed report and evidence to review the completed results without running anything.
