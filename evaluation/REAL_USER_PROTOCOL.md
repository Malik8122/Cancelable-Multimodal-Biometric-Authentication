# Real-user evaluation protocol

This protocol measures how the deployed face + voice system performs on consenting people. It uses
`evaluation/real_user_evaluation.py`, which runs the shipped enrollment and authentication code offline and writes
scores only.

**Status: no participant data has been collected.** No real-user result exists in this repository. Every real-user
metric in the paper is NOT AVAILABLE until this protocol has been run.

Before any recruitment, confirm whether your institution requires ethics or IRB approval. Recruitment is never
automated, and nothing in the repository contacts or records anyone.

## 1. Consent and anonymisation

1. Each participant gets a plain-language information sheet. It covers:
   - what is captured (face images, voice recordings);
   - that only comparison scores are kept in results;
   - how long captures are retained;
   - that they may withdraw at any time without giving a reason.
2. Written or recorded informed consent is obtained before any capture.
3. The participant is assigned an anonymous ID (`P001`, `P002`, …). The only link from an ID to a person is a paper
   or offline key held by the study lead. It is never stored in the repository or next to the captures.
4. `consent.csv` sits in the capture root and is mandatory:

   ```
   participant_id,consent,withdrawn
   P001,yes,no
   P002,yes,no
   ```

   The harness refuses to run without this file. It never reads a participant folder unless that participant's row has
   `consent=yes` and `withdrawn` is not `yes`. It also refuses any folder name that is not an anonymous ID.
5. Captures stay outside the repository, on encrypted storage. `.gitignore` excludes `data/`, but put the captures
   somewhere outside the working tree anyway.

## 2. Enrollment (per participant)

| Modality | Captures | Acceptance (unchanged product gates) |
|---|---|---|
| Face | 5 poses: `face_front`, `face_left`, `face_right`, `face_up`, `face_down` (PNG) | at least 3 poses VALID after the quality gates: confidence ≥ 0.90, sharpness ≥ 25, size ratio ≥ 0.15, centre offset ≤ 0.35, roll ≤ 20°, yaw ratio ≤ 0.20; single face |
| Voice | 2 recordings, `voice_1.wav` and `voice_2.wav` (16 kHz mono) | the consistency gate. FAIR recordings count only when `--accept-fair-voice` is passed, and the harness logs this |

Enrollment produces the product's 4 template sets per modality (T1 ACTIVE, T2–T4 STANDBY), each under its own HKDF key.
Outcomes go to `real_user_enrollment.csv`.

## 3. Genuine attempts

- At least **10 genuine face attempts** and **10 genuine voice attempts** per participant.
- Spread over at least **2 sessions on different days** where possible. Vary lighting, background and the time of day
  as they naturally occur, but do not coach towards "good" captures.
- Layout: `<root>/P001/session_1/face.png` and `voice.wav`, then `session_2/…`, and so on. Put one attempt per session
  folder. Use more session folders (`session_3`, …) to reach 10 attempts.
- `real_user_protocol_check.csv` reports, for each participant and modality, whether the minimums were met
  (`meets_protocol`). Report the numbers of participants who did and did not meet it.

## 4. Impostor attempts

No extra captures are needed. Every session capture of every participant is compared with every other participant's
enrolled templates, **under the target's key**. This is the zero-effort impostor comparison the live system would
make. Deliberate imitation, masks and replay attacks are out of scope; PAD is reported separately.

## 5. What is stored

`real_user_scores.csv` holds one row per comparison with exactly these fields:

- anonymous target ID and claimant ID
- modality
- session
- genuine/impostor label
- Hamming similarity
- estimated cosine (face) or estimated distance (voice)
- fusion score and threshold
- decision
- template set
- key version

The metrics files hold aggregates only. **No images, audio, embeddings, landmarks or templates** are written; a test
checks this (`tests/test_real_user_evaluation.py`). Templates live only in an in-memory SQLite database for the
duration of the run.

## 6. Withdrawal and deletion

```
python -m evaluation.real_user_evaluation --root <folder> --withdraw P003 --delete-captures
```

This does three things:

- removes every row in which P003 is the target or the claimant from all four output files;
- deletes `<root>/P003`;
- prints a reminder to re-run the evaluation, because the aggregates include the withdrawn participant.

Afterwards:

1. Set `withdrawn=yes` in `consent.csv`, so the participant is skipped even if their folder is restored from a backup.
2. Destroy the offline key entry for P003.
3. Delete any backups of their captures.

## 7. Analysis and reporting

- Run: `python -m evaluation.real_user_evaluation --root <folder> --consent-confirmed`
- Report per modality and for the fusion policies:
  - FAR, FRR and TAR at the configured thresholds (face estimated cosine ≥ 0.80, voice estimated distance ≤ 0.75;
    `THRESHOLD_SOURCE = TEACHER_REQUESTED_BASELINE`);
  - EER and ROC-AUC;
  - failure to acquire;
  - participant and attempt counts.
- Do **not** choose a threshold on this data and then report performance on the same data. With a small cohort, report
  the results at the configured thresholds and describe any alternative operating point as exploratory.
- Label every number **REAL DATA (n participants)** and state the cohort size and any demographic limits it implies.
